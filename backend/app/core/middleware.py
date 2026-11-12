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

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"
)


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
        self.log  = structlog.get_logger(__name__)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start       = time.perf_counter()
        request_id  = str(uuid.uuid4())[:8]
        path        = scope.get("path", "")
        client      = scope.get("client", ("unknown", 0))
        client_ip   = client[0] if client else "unknown"

        log = self.log.bind(request_id=request_id, path=path, client_ip=client_ip)

        status_code  = 500
        body_sent    = 0

        async def new_send(message):
            nonlocal status_code, body_sent
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            elif message["type"] == "http.response.body":
                body_sent += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, new_send)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info(
                "request",
                status=status_code,
                duration_ms=duration_ms,
                bytes_sent=body_sent,
            )


class PIIScrubbingMiddleware:
    """
    Scrubs PII from request bodies before they reach business logic.
    Implemented as a pure ASGI middleware to avoid BaseHTTPMiddleware streaming bugs.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_type = dict(scope.get("headers", [])).get(b"content-type", b"").decode()

        if "application/json" in content_type or "text/" in content_type:
            async def new_receive():
                message = await receive()
                if message["type"] == "http.request":
                    body    = message.get("body", b"")
                    text    = body.decode("utf-8", errors="replace")
                    scrubbed = scrub_pii(text)
                    message = {**message, "body": scrubbed.encode("utf-8")}
                return message

            await self.app(scope, new_receive, send)
        else:
            await self.app(scope, receive, send)
