"""
Production middleware stack:
1. RequestLoggingMiddleware  - structured JSON logging of every request
2. PIIScrubbingMiddleware    - masks emails, phone numbers in logs
3. RateLimitMiddleware       - per-IP rate limiting using slowapi
"""
import re
import time
import uuid

import structlog

logger = structlog.get_logger(__name__)

# Regex patterns for PII detection
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")


def scrub_pii(text: str) -> str:
    """Replace PII patterns with redacted placeholders."""
    text = _EMAIL_RE.sub("[EMAIL REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE REDACTED]", text)
    return text


class RequestLoggingMiddleware:
    """
    Logs every inbound request and its response time in structured JSON.
    Implemented as pure ASGI middleware to avoid BaseHTTPMiddleware streaming bugs.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        log = logger.bind(
            request_id=request_id,
            method=scope["method"],
            path=scope["path"],
            client_ip=client_ip,
        )

        log.info("request_received")

        status_code = 500

        async def new_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = message.get("headers", [])
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, new_send)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info("request_completed", status_code=status_code, duration_ms=duration_ms)


class PIIScrubbingMiddleware:
    """
    Scrubs PII from request bodies before they reach business logic.
    Implemented as a pure ASGI middleware to avoid BaseHTTPMiddleware streaming bugs.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Do not scrub authentication endpoints (we need the real email for login/register!)
        path = scope.get("path", "")
        if path.startswith("/api/v1/auth"):
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode("utf-8")

        if "application/json" not in content_type:
            return await self.app(scope, receive, send)

        # Buffer the entire request body
        body = b""
        more_body = True
        
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            more_body = message.get("more_body", False)

        try:
            scrubbed = scrub_pii(body.decode("utf-8"))
            new_body = scrubbed.encode("utf-8")
        except Exception:
            new_body = body

        # Generator to replay the modified body
        body_sent = False
        async def new_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": new_body, "more_body": False}
            
            return await receive()

        return await self.app(scope, new_receive, send)
