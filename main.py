import asyncio
from wikipedia_async import WikipediaClient
import wikipedia


async def main():
    # Initialize client with optimal defaults
    client = WikipediaClient()

    # Search for articles
    results = await client.search("Python programming")
    print("Search results:")
    for result in results:
        print(f"- {result.title}")

    # Get page content
    page = await client.get_page("Python (programming language)")
    print(f"Title: {page.title}")
    print(f"Summary: {page.summary[:200]}...")

    # Batch operations
    pages = await client.get_pages_batch(["Python", "JavaScript", "Rust"])
    for p in pages:
        print(f"Batch page: {p.title} - {p.summary[:100]}...")
    # Close the client session

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
