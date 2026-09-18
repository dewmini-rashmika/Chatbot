# Music Knowledge & Discovery Agent

A production-grade, multi-agent chatbot for music knowledge and discovery. Built to demonstrate enterprise AI engineering patterns.

## Architecture Highlights (CV Talking Points)

| Pattern | Implementation | Why |
|---|---|---|
| **Multi-Agent System** | LangGraph Supervisor Pattern | Specialized agents outperform one large agent |
| **Hybrid RAG** | ChromaDB (Vector) + PostgreSQL (Text-to-SQL) | Vector search hallucinate on exact facts; SQL doesn't |
| **RRF Fusion** | Reciprocal Rank Fusion | Scale-invariant result merging from two retrieval systems |
| **HITL** | LangGraph `interrupt_before` | Enterprise safety — humans approve write actions |
| **Guardrails** | Deterministic regex + LLM reviewer | Defense-in-depth: fast + nuanced safety checks |
| **Long-term Memory** | PostgreSQL fact extraction | Cross-session personalization without re-reading history |
| **Streaming** | FastAPI SSE | Real-time token streaming like ChatGPT |
| **Auth** | JWT Access + Refresh Tokens | Stateless, scalable, industry standard |
| **Middleware** | PII scrubbing + structured logging | GDPR compliance + observability |
| **Synthetic Data** | LLM-generated reviews | Sidesteps copyright — real teams do this |

## Project Structure

```
music-agent-copilot/
├── backend/
│   ├── app/
│   │   ├── agents/          # Supervisor, RAG, Tool, Reviewer agents + LangGraph graph
│   │   ├── api/v1/          # FastAPI routers: auth, conversations, chat (SSE)
│   │   ├── core/            # Config, JWT security, middlewares, dependencies
│   │   ├── db/              # Async SQLAlchemy engine
│   │   ├── guardrails/      # Input/output guardrails
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── services/        # VectorStore, SQL RAG, Long-term memory
│   │   └── tools/           # LangChain tools (MusicBrainz, Web, Spotify)
│   └── data/scripts/        # PDF ingestion, synthetic review gen, DB seeding
└── frontend/
    └── src/
        ├── components/      # Chat window, Sidebar, HITL modal
        ├── hooks/           # useChat (SSE streaming)
        ├── pages/           # Login, Register, Dashboard
        ├── services/        # Axios API client
        └── store/           # Zustand: auth + chat state
```

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL (local)

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # Add your GOOGLE_API_KEY
```

### 2. Start PostgreSQL
Make sure PostgreSQL is running on localhost:5432 and create the database:
```sql
CREATE DATABASE music_agent;
```

### 3. Run Data Pipelines
```bash
# Ingest your 68 music theory PDFs
python -m data.scripts.ingest_pdfs

# Seed structured music data (discography)
python -m data.scripts.seed_database

# Generate synthetic reviews (requires GOOGLE_API_KEY)
python -m data.scripts.generate_reviews
```

### 4. Start Backend
```bash
python -m app.main
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/api/docs
```

### 5. Frontend Setup
```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173
```

## Tech Stack
- **LLM:** Google Gemini 1.5 Flash/Pro (free tier)
- **Agent Framework:** LangGraph
- **Vector DB:** ChromaDB (local)
- **Relational DB:** PostgreSQL
- **API:** FastAPI + SSE streaming
- **Frontend:** React + Vite + Tailwind CSS + Zustand

## Data Sources
- 68 open-source music theory PDFs
- Wikipedia artist/album pages
- Synthetic reviews (AI-generated, clearly labeled)
- Seeded discography from public knowledge