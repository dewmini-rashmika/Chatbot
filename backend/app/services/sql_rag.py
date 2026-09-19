"""
SQL RAG Service — Vectorless (Structured) Retrieval via Text-to-SQL.

Why Text-to-SQL for vectorless RAG?
- Vector search cannot answer: "Which albums did Miles Davis release between 1958 and 1965?"
- SQL can answer this perfectly because it is a relational, exact-lookup query.
- The LLM translates the natural language question into SQL → we execute it → return results.

Interview talking point:
  "We chose Text-to-SQL over a static keyword index because the music schema
   is relational — you need JOINs. The LLM generates valid SQL that we
   execute against Postgres, giving the agent the same power as a human analyst."
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import asyncpg

from app.core.config import settings

_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    groq_api_key=settings.groq_api_key,
    temperature=0,
)

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
    title VARCHAR(500),
    release_year INT,
    release_date VARCHAR(50),
    label VARCHAR(255),
    genre VARCHAR(255),
    chart_positions JSONB,  -- e.g. {"US Billboard 200": 1}
    awards JSONB             -- e.g. ["Grammy Award for Best Jazz Album 1961"]
)

TABLE tracks (
    id UUID PRIMARY KEY,
    album_id UUID REFERENCES albums(id),
    title VARCHAR(500),
    track_number INT,
    duration_ms INT
)
"""

TEXT_TO_SQL_PROMPT = f"""You are a SQL expert for a music knowledge database.
Convert the user's question into a valid PostgreSQL SELECT query.

{SCHEMA_DESCRIPTION}

Rules:
- Return ONLY the SQL query, no explanation.
- Always use LIMIT 20 to prevent huge result sets.
- Use ILIKE for case-insensitive text matching.
- Only generate SELECT statements — never INSERT, UPDATE, or DELETE.
- If the question cannot be answered by SQL, return: SELECT 'NOT_SQL_QUERY' as result;
"""


class SQLRAGService:
    """
    Translates natural language queries to SQL and executes them
    against the PostgreSQL music knowledge schema.
    """

    async def search(self, query: str) -> list[dict]:
        """
        Main entry: generate SQL from query, execute it, return results.
        """
        sql = await self._generate_sql(query)

        # Safety check — only allow SELECT
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT") or "NOT_SQL_QUERY" in sql_upper:
            return []

        results = await self._execute_sql(sql)
        return results

    async def _generate_sql(self, query: str) -> str:
        """Use Gemini to translate natural language → SQL."""
        messages = [
            SystemMessage(content=TEXT_TO_SQL_PROMPT),
            HumanMessage(content=f"Question: {query}"),
        ]
        response = await _llm.ainvoke(messages)
        # Strip markdown code fences if present
        sql = response.content.strip().strip("```sql").strip("```").strip()
        return sql

    async def _execute_sql(self, sql: str) -> list[dict]:
        """Execute the generated SQL query and return rows as dicts."""
        try:
            conn = await asyncpg.connect(settings.sync_database_url.replace(
                "postgresql://", "postgresql://"
            ))
            rows = await conn.fetch(sql)
            await conn.close()

            results = []
            for i, row in enumerate(rows):
                row_dict = dict(row)
                content = ", ".join(f"{k}: {v}" for k, v in row_dict.items() if v is not None)
                results.append({
                    "id": f"sql_{hash(content)}",
                    "content": content,
                    "source": "PostgreSQL Music Database",
                    "doc_type": "structured",
                    "retrieval_method": "sql",
                    "raw": row_dict,
                })
            return results

        except Exception as e:
            # Never let a SQL error crash the agent — return empty results
            return []
