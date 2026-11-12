"""
SQL RAG Service — Vectorless (Structured) Retrieval via Text-to-SQL.

Why Text-to-SQL for vectorless RAG?
- Vector search cannot answer: "Which albums did Miles Davis release between 1958 and 1965?"
- SQL can answer this perfectly because it is a relational, exact-lookup query.
- The LLM translates the natural-language question into a PostgreSQL SELECT statement.

LLM choice: groq/llama-3.3-70b-versatile (via LiteLLM)
  - Strongest at deterministic syntax, JOIN operations, and avoiding SQL errors.
  - Swap to "anthropic/claude-3-5-haiku-20241022" or "deepseek/deepseek-coder"
    via SQL_MODEL env var for even higher SQL accuracy.
"""

import re
import asyncpg
import structlog
from app.core.llm_factory import call_llm, get_model_for_role
from app.core.config import settings

logger = structlog.get_logger(__name__)

SCHEMA_DESCRIPTION = """
Database schema for music knowledge:

TABLE artists (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    birth_year INT,
    death_year INT,
    nationality VARCHAR(100),
    genres JSONB         -- e.g. ["jazz", "bebop"]
)

TABLE albums (
    id UUID PRIMARY KEY,
    artist_id UUID REFERENCES artists(id),
    title VARCHAR(255),
    release_year INT,
    genre VARCHAR(100),
    label VARCHAR(255)
)

TABLE tracks (
    id UUID PRIMARY KEY,
    album_id UUID REFERENCES albums(id),
    artist_id UUID REFERENCES artists(id),
    title VARCHAR(255),
    duration_seconds INT,
    track_number INT
)

TABLE chart_positions (
    id UUID PRIMARY KEY,
    track_id UUID REFERENCES tracks(id),
    chart_name VARCHAR(255),
    position INT,
    chart_date DATE,
    country VARCHAR(10)
)

TABLE music_reviews (
    id UUID PRIMARY KEY,
    album_id UUID REFERENCES albums(id),
    reviewer VARCHAR(255),
    score FLOAT,
    review_text TEXT,
    review_date DATE
)

TABLE genres (
    id UUID PRIMARY KEY,
    name VARCHAR(100),
    parent_genre VARCHAR(100),
    description TEXT
)
"""

TEXT_TO_SQL_PROMPT = (
    "You are a SQL expert for a music knowledge database.\n"
    "Convert the user's question into a valid PostgreSQL SELECT query.\n\n"
    + SCHEMA_DESCRIPTION
    + """
Rules:
- Return ONLY the SQL query, no explanation.
- Always use LIMIT 20 to prevent huge result sets.
- Use ILIKE for case-insensitive text matching.
- Only generate SELECT statements — never INSERT, UPDATE, or DELETE.
- If the question cannot be answered by SQL, return: SELECT 'NOT_SQL_QUERY' as result;
"""
)


class SQLRAGService:
    """Translates natural language queries to SQL and executes them
    against the PostgreSQL music knowledge schema."""

    async def search(self, query: str) -> list[dict]:
        """
        Main entry: generate SQL from query, execute it, return results.
        Returns an empty list if the query is not SQL-answerable.
        """
        try:
            sql = await self._generate_sql(query)
            if not sql or "NOT_SQL_QUERY" in sql:
                return []
            return await self._execute_sql(sql)
        except Exception as exc:
            logger.warning("sql_rag.search_failed", error=str(exc), query=query[:80])
            return []

    async def _generate_sql(self, query: str) -> str:
        """Use LiteLLM to translate natural language → SQL."""
        model = get_model_for_role("sql")
        logger.info("sql_rag.generating", query=query[:80], model=model)

        messages = [
            {"role": "system", "content": TEXT_TO_SQL_PROMPT},
            {"role": "user",   "content": f"Question: {query}"},
        ]

        response = await call_llm(
            role="sql",
            messages=messages,
            temperature=0.0,   # deterministic — we want exact SQL
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip ```sql ... ``` fences if present
        raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        sql = raw.strip()

        # Safety guard — reject any non-SELECT statements
        sql_upper = sql.upper().lstrip()
        if not sql_upper.startswith("SELECT"):
            logger.warning("sql_rag.non_select_blocked", sql=sql[:100])
            return ""

        return sql

    async def _execute_sql(self, sql: str) -> list[dict]:
        """Execute the generated SQL query and return rows as dicts."""
        # asyncpg needs a plain postgresql:// URL (not +asyncpg dialect prefix)
        db_url = settings.sync_database_url
        if db_url.startswith("postgresql+"):
            db_url = "postgresql" + db_url[len("postgresql+asyncpg"):]

        conn = await asyncpg.connect(db_url)
        try:
            rows = await conn.fetch(sql)
            results = []
            for i, row in enumerate(rows):
                row_dict = {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                            for k, v in dict(row).items()}
                row_dict["_rank"] = i
                row_dict["_source"] = "PostgreSQL Music Database"
                results.append(row_dict)
            return results
        finally:
            await conn.close()
