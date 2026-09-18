"""
Synthetic Music Review Generator.
Uses Gemini to generate 30 synthetic reviews styled after different critical voices.
These are clearly labeled as synthetic in the metadata and README.

Why synthetic data?
  Real professional reviews (Pitchfork, Rolling Stone) are copyrighted.
  Real teams generate synthetic corpora for licensing reasons —
  this is a legitimate, defensible engineering decision.

Run: python -m data.scripts.generate_reviews
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings
from app.services.vector_store import VectorStoreService

_llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=0.8,  # Higher temp for creative variety
)

# Artists and albums to generate reviews for
REVIEW_TARGETS = [
    {"artist": "Miles Davis", "album": "Kind of Blue", "year": 1959, "genre": "Modal Jazz"},
    {"artist": "Miles Davis", "album": "Bitches Brew", "year": 1970, "genre": "Jazz Fusion"},
    {"artist": "John Coltrane", "album": "A Love Supreme", "year": 1965, "genre": "Avant-garde Jazz"},
    {"artist": "The Beatles", "album": "Revolver", "year": 1966, "genre": "Psychedelic Rock"},
    {"artist": "The Beatles", "album": "Sgt. Pepper's", "year": 1967, "genre": "Art Rock"},
    {"artist": "Pink Floyd", "album": "The Dark Side of the Moon", "year": 1973, "genre": "Progressive Rock"},
    {"artist": "David Bowie", "album": "Heroes", "year": 1977, "genre": "Art Rock"},
    {"artist": "Prince", "album": "Purple Rain", "year": 1984, "genre": "Funk/Rock"},
    {"artist": "Kendrick Lamar", "album": "To Pimp a Butterfly", "year": 2015, "genre": "Hip-Hop"},
    {"artist": "Björk", "album": "Homogenic", "year": 1997, "genre": "Electronic/Art Pop"},
]

CRITICAL_VOICES = [
    "academic musicologist focusing on structure and form",
    "music journalist with deep jazz knowledge writing for a culture magazine",
    "production-focused critic analyzing sonic textures and studio techniques",
]

REVIEW_PROMPT = """Write a music review in the style of a {voice}.
Album: {album} by {artist} ({year}) — Genre: {genre}

The review should be 3-4 paragraphs, approximately 300-400 words.
Focus on musical substance: harmony, rhythm, production, cultural context.
DO NOT reproduce any song lyrics.
Clearly label this as a critical analysis.
"""


async def generate_reviews():
    """Generate synthetic reviews and ingest them into ChromaDB."""
    service = VectorStoreService()
    all_reviews = []
    total_chunks = 0

    print(f"🎵 Generating {len(REVIEW_TARGETS) * len(CRITICAL_VOICES)} synthetic reviews...\n")

    for target in REVIEW_TARGETS:
        for voice in CRITICAL_VOICES:
            prompt = REVIEW_PROMPT.format(voice=voice, **target)
            print(f"  ✍  {target['artist']} - {target['album']} [{voice[:30]}...]", end=" ")

            try:
                response = await _llm.ainvoke([
                    SystemMessage(content="You are a music critic. Generate substantive reviews without reproducing lyrics."),
                    HumanMessage(content=prompt),
                ])
                review_text = response.content

                # Prepend clear synthetic label
                labeled_review = (
                    f"[SYNTHETIC REVIEW — AI-generated for research/demo purposes]\n\n"
                    f"Album: {target['album']} by {target['artist']} ({target['year']})\n"
                    f"Genre: {target['genre']}\n\n"
                    f"{review_text}"
                )

                chunks = await service.ingest_text(
                    text=labeled_review,
                    source=f"synthetic_review_{target['artist'].replace(' ', '_')}_{target['album'].replace(' ', '_')}",
                    doc_type="synthetic_review",
                )
                total_chunks += chunks
                all_reviews.append({**target, "review": review_text[:200] + "..."})
                print(f"✅ ({chunks} chunks)")

            except Exception as e:
                print(f"❌ {e}")

            await asyncio.sleep(1)  # Respect API rate limits

    # Save a manifest
    manifest_path = Path(__file__).parent.parent / "synthetic_reviews_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_reviews, f, indent=2)

    print(f"\n✅ Generated {len(all_reviews)} reviews → {total_chunks} chunks in ChromaDB")
    print(f"📄 Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(generate_reviews())
