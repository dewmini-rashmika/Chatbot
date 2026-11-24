"""
LangSmith RAG Evaluation — Music Knowledge & Discovery Agent
=============================================================

What this file does:
  1. Creates a LangSmith dataset of 20 Q&A pairs drawn from your actual PDFs
  2. Runs your REAL RAG pipeline (ChromaDB + PostgreSQL RRF + Gemini) on them
  3. Scores every answer with 3 evaluators (correctness, relevance, faithfulness)
  4. Prints a numeric results table in the terminal
  5. Saves all results to LangSmith UI for visual comparison across runs

Run (from backend/):
    python -m tests.langsmith_eval
    python -m tests.langsmith_eval --experiment music-rag-v2-larger-chunks
    python -m tests.langsmith_eval --compare music-rag-baseline music-rag-v2-larger-chunks

Requirements already in requirements.txt:
    langsmith==0.2.11
    langchain==0.3.12
    langchain-community==0.3.12

You need ONE new env var added to .env:
    LANGCHAIN_API_KEY=lsv2_pt_your_key_here   ← get from smith.langchain.com
    LANGCHAIN_TRACING_V2=true                  ← already in .env.example, just flip to true
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# ── Make sure backend/ is on sys.path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Use SelectorEventLoop on Windows to avoid Proactor "Event loop is closed" errors
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── LangSmith ─────────────────────────────────────────────────────────────────
from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Run, Example

# ── Your actual app services (no mocking — real pipeline) ────────────────────
from app.services.vector_store import VectorStoreService
from app.services.sql_rag import SQLRAGService
from app.agents.rag_agent import reciprocal_rank_fusion, RAG_SYSTEM_PROMPT
from app.core.llm_factory import call_llm, get_model_for_role


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET
# 20 questions written from your actual PDF topics (01xx–04xx, SCxx, SPxx)
# Each has a "ground_truth" answer the evaluator scores against.
# ══════════════════════════════════════════════════════════════════════════════

DATASET_NAME = "music-rag-eval-v1"

EVAL_EXAMPLES = [
    # ── Fundamentals (PDF 01xx) ───────────────────────────────────────────────
    {
        "question": "What is music theory and why is it important?",
        "ground_truth": (
            "Music theory is the study of the practices and possibilities of music. "
            "It provides a framework for understanding how music works, including pitch, "
            "rhythm, scales, harmony, and form."
        ),
    },
    {
        "question": "How is a major scale constructed?",
        "ground_truth": (
            "A major scale follows the interval pattern whole-whole-half-whole-whole-whole-half "
            "(W-W-H-W-W-W-H). For example, C major uses the notes C D E F G A B."
        ),
    },
    {
        "question": "What is the circle of fifths?",
        "ground_truth": (
            "The circle of fifths is a visual arrangement of all 12 keys where each key "
            "is a perfect fifth above the previous. It shows key signature relationships "
            "and aids in modulation and chord progression planning."
        ),
    },
    {
        "question": "What is a key signature?",
        "ground_truth": (
            "A key signature is a set of sharps or flats written at the start of a staff "
            "that indicates which notes are consistently altered throughout a piece."
        ),
    },
    {
        "question": "What is the difference between perfect and imperfect intervals?",
        "ground_truth": (
            "Perfect intervals (unison, 4th, 5th, octave) are maximally stable and have "
            "only one diatonic form. Imperfect intervals (3rds and 6ths) come in major "
            "and minor versions and are considered consonant but less stable."
        ),
    },
    # ── Minor Scales & Triads (PDF 0111–0115) ────────────────────────────────
    {
        "question": "What are the three forms of the minor scale?",
        "ground_truth": (
            "Natural minor uses only diatonic notes. Harmonic minor raises the 7th degree. "
            "Melodic minor raises both the 6th and 7th degrees ascending, then reverts "
            "to natural minor descending."
        ),
    },
    {
        "question": "What is a triad and what are its three basic types?",
        "ground_truth": (
            "A triad is a three-note chord built in thirds. Major triads use a major 3rd "
            "below a minor 3rd. Minor triads use a minor 3rd below a major 3rd. "
            "Diminished triads use two minor 3rds."
        ),
    },
    {
        "question": "What is figured bass notation?",
        "ground_truth": (
            "Figured bass uses numbers below a bass note to indicate intervals and chord "
            "inversions to be played above it. It originated in Baroque music and is used "
            "in theory to specify chord voicings, e.g. '6' means first inversion."
        ),
    },
    # ── Part Writing (PDF 02xx) ───────────────────────────────────────────────
    {
        "question": "What is voice leading (part writing) in music theory?",
        "ground_truth": (
            "Voice leading is the technique of writing independent melodic lines that "
            "harmonize smoothly. Rules prohibit parallel perfect intervals, voice crossing, "
            "and large awkward leaps, favouring stepwise motion."
        ),
    },
    {
        "question": "What are the four main types of harmonic cadences?",
        "ground_truth": (
            "Authentic cadence (V–I) is the strongest resolution. Half cadence ends on V. "
            "Plagal cadence is IV–I. Deceptive cadence moves V to vi, avoiding the "
            "expected tonic."
        ),
    },
    {
        "question": "What is diatonic common-chord modulation?",
        "ground_truth": (
            "Diatonic common-chord modulation changes keys using a pivot chord that belongs "
            "diatonically to both the original and destination key, creating a smooth "
            "harmonic transition."
        ),
    },
    # ── Harmony (PDF 03xx) ────────────────────────────────────────────────────
    {
        "question": "What are secondary dominants?",
        "ground_truth": (
            "Secondary dominants are dominant-seventh chords built on scale degrees other "
            "than 5̂, used to briefly tonicize a chord other than the tonic. For example, "
            "V7/V is a secondary dominant that makes the V chord feel like a local tonic."
        ),
    },
    {
        "question": "What is the Neapolitan chord?",
        "ground_truth": (
            "The Neapolitan chord (N6 or ♭II6) is a major chord built on the flattened "
            "second scale degree, typically in first inversion. It functions as a dramatic "
            "predominant chord preceding the dominant."
        ),
    },
    {
        "question": "What are augmented sixth chords and what are their three types?",
        "ground_truth": (
            "Augmented sixth chords are chromatic harmonies featuring an augmented sixth "
            "interval between ♭6̂ and #4̂. The three types are Italian (It+6), "
            "French (Fr+6), and German (Gr+6) augmented sixth chords, each adding "
            "different inner voices."
        ),
    },
    # ── Modern Theory (PDF 04xx) ──────────────────────────────────────────────
    {
        "question": "What are the seven modern modes and how do they relate to the major scale?",
        "ground_truth": (
            "The seven modes—Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, "
            "Locrian—are derived by starting a major scale on each of its seven degrees. "
            "Each mode has a unique pattern of whole and half steps and a distinct character."
        ),
    },
    {
        "question": "What is atonality in 20th-century music?",
        "ground_truth": (
            "Atonality avoids a tonal centre or key, treating all 12 pitches as equal. "
            "It emerged in the early 20th century, notably in works by Schoenberg, Berg, "
            "and Webern, as a rejection of functional tonal harmony."
        ),
    },
    {
        "question": "What is twelve-tone serialism?",
        "ground_truth": (
            "Twelve-tone serialism, invented by Schoenberg, organises all 12 chromatic "
            "pitches into a fixed tone row. Compositions derive material from the row and "
            "its transformations: inversion, retrograde, and retrograde-inversion."
        ),
    },
    # ── Species Counterpoint (PDF SCxx) ──────────────────────────────────────
    {
        "question": "What is species counterpoint?",
        "ground_truth": (
            "Species counterpoint is a pedagogical system for learning multi-voice writing "
            "in five progressive stages: 1st (note-against-note), 2nd (two notes per beat), "
            "3rd (four notes per beat), 4th (syncopations), and 5th species (florid, "
            "combining all previous techniques)."
        ),
    },
    {
        "question": "How are consonances and dissonances treated differently in counterpoint?",
        "ground_truth": (
            "Consonances (perfect: unison, 5th, octave; imperfect: 3rds and 6ths) may "
            "appear on strong beats without restriction. Dissonances (2nds, 4ths against "
            "bass, 7ths, tritone) are unstable and must be introduced and resolved by "
            "stepwise motion."
        ),
    },
    # ── Theory Exercises (PDF SPxx) ──────────────────────────────────────────
    {
        "question": "What are suspensions in music and how do they work?",
        "ground_truth": (
            "A suspension is a non-harmonic tone where a note from the previous chord is "
            "held over into the next chord, creating dissonance, before resolving downward "
            "by step to a chord tone. Common suspensions are 4-3, 7-6, and 9-8."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CREATE DATASET IN LANGSMITH
# ══════════════════════════════════════════════════════════════════════════════

def create_or_update_dataset(client: Client) -> str:
    """
    Creates the LangSmith dataset once; reuses it on subsequent runs.
    Returns the dataset name string.
    """
    existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if existing:
        print(f"  ✅ Dataset '{DATASET_NAME}' already exists — reusing it.")
        return DATASET_NAME

    print(f"  📦 Creating dataset '{DATASET_NAME}' ({len(EVAL_EXAMPLES)} examples)...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Music Theory RAG evaluation dataset — 20 Q&A pairs "
            "drawn from PDF topics: fundamentals, part writing, harmony, "
            "modern theory, counterpoint, and exercises."
        ),
    )
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in EVAL_EXAMPLES],
        outputs=[{"answer": ex["ground_truth"]} for ex in EVAL_EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"  ✅ Dataset created with {len(EVAL_EXAMPLES)} Q&A pairs.")
    return DATASET_NAME


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — YOUR REAL RAG PIPELINE (called once per dataset example)
# ══════════════════════════════════════════════════════════════════════════════

async def _rag_async(question: str) -> str:
    """
    Runs the ACTUAL hybrid RAG pipeline:
      ChromaDB semantic search  ─┐
                                 ├─ RRF fusion ─► Gemini answer
      PostgreSQL SQL search     ─┘

    This is exactly what rag_agent_node() does — no mocking.
    """
    vector_svc = VectorStoreService()
    sql_svc    = SQLRAGService()

    vector_results, sql_results = await asyncio.gather(
        vector_svc.search(question),
        sql_svc.search(question),
    )

    fused = reciprocal_rank_fusion(vector_results, sql_results)

    context_parts = []
    for doc in fused[:10]:
        content = doc.get("content", doc.get("document", str(doc)))
        source  = doc.get("source", "Internal Database")
        if len(content) > 1500:
            content = content[:1500] + "... [TRUNCATED]"
        context_parts.append(f"[{source}]\n{content}")

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No results found."

    llm_messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {question}",
        },
    ]

    response = await call_llm(role="rag", messages=llm_messages, temperature=0.3)
    return response.choices[0].message.content


def _safe_run_async(coro):
    """Safely run an async coroutine across threads without Windows Proactor loop closure errors."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def rag_pipeline(inputs: dict) -> dict:
    """
    Sync wrapper — LangSmith `evaluate()` calls this per example.
    Returns {"answer": "..."} so evaluators can compare with ground truth.
    """
    question = inputs["question"]
    answer   = _safe_run_async(_rag_async(question))
    return {"answer": answer}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATORS (3 LLM-as-judge scorers, each returns 0.0–1.0)
# ══════════════════════════════════════════════════════════════════════════════

def _judge_prompt(criterion: str, question: str, answer: str, reference: str = "") -> str:
    ref_block = f"\nReference answer:\n{reference}" if reference else ""
    return (
        f"You are an impartial judge evaluating a music theory RAG system.\n\n"
        f"Question: {question}\n"
        f"System answer: {answer}"
        f"{ref_block}\n\n"
        f"Criterion: {criterion}\n\n"
        f"Score the answer on this criterion from 0 to 1 (decimals allowed).\n"
        f"Respond with ONLY a JSON object: {{\"score\": <float>, \"reason\": \"<one sentence>\"}}"
    )


async def _llm_judge(criterion: str, question: str, answer: str, reference: str = "") -> float:
    """Calls the reviewer LLM as a judge. Returns a score 0.0–1.0."""
    prompt = _judge_prompt(criterion, question, answer, reference)
    try:
        response = await call_llm(
            role="reviewer",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = raw.strip("```json").strip("```").strip()
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as exc:
        print(f"    ⚠ Judge call failed: {exc}")
        return 0.0


def correctness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Is the answer factually correct compared to the ground-truth reference?
    Score: 0.0 (wrong) → 1.0 (fully correct)
    """
    answer    = (run.outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("answer", "")
    question  = (example.inputs or {}).get("question", "")

    criterion = (
        "Factual correctness: Does the system answer contain the same key facts "
        "and music theory concepts as the reference answer?"
    )
    score = _safe_run_async(_llm_judge(criterion, question, answer, reference))
    return EvaluationResult(key="correctness", score=score)


def relevance_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Does the answer directly address the question asked?
    Score: 0.0 (off-topic) → 1.0 (directly relevant)
    """
    answer   = (run.outputs or {}).get("answer", "")
    question = (example.inputs or {}).get("question", "")

    criterion = (
        "Relevance: Does the answer directly address the question? "
        "Penalise answers that go off-topic or answer a different question."
    )
    score = _safe_run_async(_llm_judge(criterion, question, answer))
    return EvaluationResult(key="relevance", score=score)


def faithfulness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Does the answer stay grounded in music theory facts (no hallucinations)?
    Score: 0.0 (many hallucinations) → 1.0 (fully grounded)
    """
    answer   = (run.outputs or {}).get("answer", "")
    question = (example.inputs or {}).get("question", "")

    criterion = (
        "Faithfulness / no hallucination: Does the answer avoid inventing facts, "
        "artist names, dates, or chord labels not supported by standard music theory? "
        "Penalise any factually invented content."
    )
    score = _safe_run_async(_llm_judge(criterion, question, answer))
    return EvaluationResult(key="faithfulness", score=score)


EVALUATORS = [correctness_evaluator, relevance_evaluator, faithfulness_evaluator]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RUN EVALUATION & PRINT NUMERIC RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_scores(results) -> dict[str, float]:
    """Extract per-metric average scores from an evaluate() result object."""
    scores: dict[str, list[float]] = {}
    for result in results:
        feedbacks = result.get("evaluation_results", {}).get("results", [])
        for fb in feedbacks:
            key = fb.key
            val = fb.score
            if val is not None:
                scores.setdefault(key, []).append(float(val))
    return {k: round(sum(v) / len(v), 4) for k, v in scores.items() if v}


def _print_table(avg: dict[str, float], experiment: str):
    print(f"\n{'═'*55}")
    print(f"  RESULTS  ›  {experiment}")
    print(f"{'═'*55}")
    print(f"  {'Metric':<20}  {'Avg Score':>10}  (0.0 = worst, 1.0 = best)")
    print(f"  {'-'*48}")
    for metric, score in sorted(avg.items()):
        bar = "█" * int(score * 20)
        print(f"  {metric:<20}  {score:>10.4f}  {bar}")
    print(f"{'═'*55}\n")


def run_evaluation(experiment_prefix: str = "music-rag-baseline") -> dict[str, float]:
    """
    Main entry point.  Creates dataset → runs pipeline → scores → prints table.
    Returns average scores dict for optional cross-run comparison.
    """
    # ── Validate env ──────────────────────────────────────────────────────────
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not langchain_api_key or langchain_api_key.startswith("lsv2_pt_your"):
        print("\n❌  LANGCHAIN_API_KEY is not set or still placeholder.")
        print("    Add your real key to backend/.env and re-run.\n")
        return {}

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "music-agent")
    os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    client = Client()

    print(f"\n🎵  Music Theory RAG — LangSmith Evaluation")
    print(f"    Vector DB : ChromaDB  (embedding: all-MiniLM-L6-v2)")
    print(f"    SQL DB    : PostgreSQL (Text-to-SQL via {os.getenv('SQL_MODEL','groq/llama-3.3-70b-versatile')})")
    print(f"    Fusion    : RRF k=60")
    print(f"    Generator : {os.getenv('RAG_MODEL','gemini/gemini-1.5-flash')}")
    print(f"    Judge     : {os.getenv('REVIEWER_MODEL','groq/llama-3.3-70b-versatile')}")
    print()

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    dataset_name = create_or_update_dataset(client)

    # ── Step 2: Run ───────────────────────────────────────────────────────────
    print(f"  🚀 Running experiment: '{experiment_prefix}'")
    print(f"     (Calling real RAG pipeline on all {len(EVAL_EXAMPLES)} examples…)\n")

    results = evaluate(
        rag_pipeline,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix=experiment_prefix,
        metadata={
            "rag_version":     experiment_prefix,
            "vector_db":       "chromadb",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size":      800,
            "chunk_overlap":   100,
            "top_k":           10,
            "fusion":          "RRF-k60",
            "rag_llm":         os.getenv("RAG_MODEL", "gemini/gemini-1.5-flash"),
            "judge_llm":       os.getenv("REVIEWER_MODEL", "groq/llama-3.3-70b-versatile"),
        },
    )

    # ── Step 3: Print ─────────────────────────────────────────────────────────
    avg_scores = _aggregate_scores(results)
    _print_table(avg_scores, experiment_prefix)

    project = os.getenv("LANGCHAIN_PROJECT", "music-agent")
    print(f"  📊  Full run visible at: https://smith.langchain.com/projects/{project}\n")

    return avg_scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — COMPARE TWO EXPERIMENTS (optional)
# ══════════════════════════════════════════════════════════════════════════════

def compare_experiments(
    scores_a: dict[str, float],
    scores_b: dict[str, float],
    name_a: str,
    name_b: str,
):
    """Prints a side-by-side delta table so you can see numeric improvement."""
    print(f"\n{'═'*70}")
    print(f"  COMPARISON  ›  {name_a}   vs   {name_b}")
    print(f"{'═'*70}")
    col_a = name_a[:18]
    col_b = name_b[:18]
    print(f"  {'Metric':<20}  {col_a:>18}  {col_b:>18}  {'Delta':>8}")
    print(f"  {'-'*66}")

    all_metrics = sorted(set(list(scores_a) + list(scores_b)))
    for metric in all_metrics:
        sa = scores_a.get(metric, 0.0)
        sb = scores_b.get(metric, 0.0)
        delta = sb - sa
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
        print(f"  {metric:<20}  {sa:>18.4f}  {sb:>18.4f}  {arrow} {delta:>+7.4f}")

    print(f"{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluation on the Music Theory RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline run
  python -m tests.langsmith_eval

  # Named experiment (e.g. after changing chunk_size in vector_store.py)
  python -m tests.langsmith_eval --experiment music-rag-v2-chunk1000

  # Run baseline AND a second experiment, then compare
  python -m tests.langsmith_eval --compare music-rag-baseline music-rag-v2-chunk1000
""",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="music-rag-baseline",
        help="Name for this experiment run (default: music-rag-baseline)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("EXPERIMENT_A", "EXPERIMENT_B"),
        default=None,
        help="Run two experiments sequentially, then print a delta comparison table",
    )
    args = parser.parse_args()

    if args.compare:
        name_a, name_b = args.compare
        print(f"\n📋  Running TWO experiments and comparing: '{name_a}'  vs  '{name_b}'")
        scores_a = run_evaluation(name_a)
        scores_b = run_evaluation(name_b)
        if scores_a and scores_b:
            compare_experiments(scores_a, scores_b, name_a, name_b)
    else:
        run_evaluation(args.experiment)
