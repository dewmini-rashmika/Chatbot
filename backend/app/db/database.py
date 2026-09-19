"""
SQLAlchemy async database engine + session factory.
Using AsyncSession for non-blocking DB operations throughout the app.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# Async engine — connects to PostgreSQL via asyncpg driver
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",  # Log SQL only in dev
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Validate connections before use
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy models."""
    pass


async def get_db() -> AsyncSession:
    """
    FastAPI dependency that yields a DB session per request.
    Ensures the session is always closed, even on exception.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
