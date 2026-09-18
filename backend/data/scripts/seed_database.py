"""
PostgreSQL Music Knowledge Schema Seed Script.
Creates and populates the structured music data used for Vectorless (Text-to-SQL) RAG.

Run: python -m data.scripts.seed_database
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings
from app.models.models import Base, Artist, Album, Track

SEED_DATA = [
    {
        "artist": {
            "name": "Miles Davis",
            "birth_year": 1926,
            "death_year": 1991,
            "nationality": "American",
            "genres": ["jazz", "modal jazz", "jazz fusion", "bebop"],
        },
        "albums": [
            {
                "title": "Kind of Blue",
                "release_year": 1959,
                "release_date": "1959-08-17",
                "label": "Columbia Records",
                "genre": "Modal Jazz",
                "chart_positions": {"US Billboard 200": None},
                "awards": ["Grammy Hall of Fame 1992", "RIAA Platinum"],
                "tracks": [
                    "So What", "Freddie Freeloader", "Blue in Green",
                    "All Blues", "Flamenco Sketches"
                ],
            },
            {
                "title": "Bitches Brew",
                "release_year": 1970,
                "release_date": "1970-03-30",
                "label": "Columbia Records",
                "genre": "Jazz Fusion",
                "chart_positions": {"US Billboard 200": 35},
                "awards": ["Grammy Award for Best Jazz Large Ensemble Album 1971"],
                "tracks": ["Pharaoh's Dance", "Bitches Brew", "Spanish Key", "John McLaughlin"],
            },
            {
                "title": "Birth of the Cool",
                "release_year": 1957,
                "release_date": "1957-01-01",
                "label": "Capitol Records",
                "genre": "Cool Jazz",
                "chart_positions": {},
                "awards": [],
                "tracks": ["Move", "Jeru", "Moon Dreams", "Venus de Milo", "Budo"],
            },
        ],
    },
    {
        "artist": {
            "name": "John Coltrane",
            "birth_year": 1926,
            "death_year": 1967,
            "nationality": "American",
            "genres": ["jazz", "hard bop", "avant-garde jazz", "modal jazz"],
        },
        "albums": [
            {
                "title": "A Love Supreme",
                "release_year": 1965,
                "release_date": "1965-01-28",
                "label": "Impulse! Records",
                "genre": "Avant-garde Jazz",
                "chart_positions": {},
                "awards": ["Grammy Hall of Fame 1999"],
                "tracks": ["Part I: Acknowledgement", "Part II: Resolution",
                           "Part III: Pursuance", "Part IV: Psalm"],
            },
            {
                "title": "Blue Train",
                "release_year": 1957,
                "release_date": "1957-09-15",
                "label": "Blue Note Records",
                "genre": "Hard Bop",
                "chart_positions": {},
                "awards": [],
                "tracks": ["Blue Train", "Moment's Notice", "Locomotion", "I'm Old Fashioned"],
            },
        ],
    },
    {
        "artist": {
            "name": "The Beatles",
            "birth_year": 1960,
            "death_year": 1970,
            "nationality": "British",
            "genres": ["rock", "pop", "psychedelic rock", "art rock"],
        },
        "albums": [
            {
                "title": "Revolver",
                "release_year": 1966,
                "release_date": "1966-08-05",
                "label": "Parlophone",
                "genre": "Psychedelic Rock",
                "chart_positions": {"UK Albums Chart": 1, "US Billboard 200": 1},
                "awards": ["Grammy Hall of Fame 2013"],
                "tracks": ["Taxman", "Eleanor Rigby", "Love You To", "Yellow Submarine",
                           "Good Day Sunshine", "Here, There and Everywhere"],
            },
            {
                "title": "Sgt. Pepper's Lonely Hearts Club Band",
                "release_year": 1967,
                "release_date": "1967-06-01",
                "label": "Parlophone",
                "genre": "Art Rock",
                "chart_positions": {"UK Albums Chart": 1, "US Billboard 200": 1},
                "awards": ["Grammy Award for Album of the Year 1968",
                           "Grammy Award for Best Contemporary Album 1968"],
                "tracks": ["Sgt. Pepper's Lonely Hearts Club Band", "With a Little Help from My Friends",
                           "Lucy in the Sky with Diamonds", "A Day in the Life"],
            },
        ],
    },
    {
        "artist": {
            "name": "Kendrick Lamar",
            "birth_year": 1987,
            "nationality": "American",
            "genres": ["hip-hop", "rap", "conscious hip-hop", "jazz rap"],
        },
        "albums": [
            {
                "title": "To Pimp a Butterfly",
                "release_year": 2015,
                "release_date": "2015-03-15",
                "label": "Top Dawg Entertainment / Aftermath / Interscope",
                "genre": "Hip-Hop / Jazz",
                "chart_positions": {"US Billboard 200": 1},
                "awards": ["Grammy Award for Best Rap Album 2016",
                           "Grammy Award for Best Rap Song 2016"],
                "tracks": ["Wesley's Theory", "King Kunta", "Alright",
                           "The Blacker the Berry", "Mortal Man"],
            },
            {
                "title": "DAMN.",
                "release_year": 2017,
                "release_date": "2017-04-14",
                "label": "Top Dawg Entertainment / Aftermath / Interscope",
                "genre": "Hip-Hop",
                "chart_positions": {"US Billboard 200": 1},
                "awards": ["Pulitzer Prize for Music 2018",
                           "Grammy Award for Best Rap Album 2018"],
                "tracks": ["BLOOD.", "DNA.", "ELEMENT.", "HUMBLE.", "LOVE."],
            },
        ],
    },
]


async def seed_database():
    """Seed the PostgreSQL database with structured music knowledge."""
    engine = create_async_engine(settings.database_url, echo=False)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("🎵 Seeding PostgreSQL music knowledge database...\n")

    async with SessionLocal() as session:
        for entry in SEED_DATA:
            artist_data = entry["artist"]
            print(f"  Adding artist: {artist_data['name']}")

            artist = Artist(**artist_data)
            session.add(artist)
            await session.flush()

            for album_data in entry["albums"]:
                tracks = album_data.pop("tracks", [])
                album = Album(artist_id=artist.id, **album_data)
                session.add(album)
                await session.flush()

                for i, track_name in enumerate(tracks, 1):
                    track = Track(album_id=album.id, title=track_name, track_number=i)
                    session.add(track)

                print(f"    ✅ {album.title} ({album.release_year}) — {len(tracks)} tracks")

        await session.commit()

    print(f"\n✅ Database seeded with {len(SEED_DATA)} artists!")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed_database())
