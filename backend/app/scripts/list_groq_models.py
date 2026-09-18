import asyncio
import os
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

async def list_models():
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
    models = await client.models.list()
    for m in models.data:
        print(m.id)

if __name__ == "__main__":
    asyncio.run(list_models())
