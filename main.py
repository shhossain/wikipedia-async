import asyncio
from wikipedia_async import WikipediaClient
from pprint import pprint


async def main():
    # Initialize client with optimal defaults
    client = WikipediaClient()

    # Search for articles
    results = await client.search("python", suggestion=True)
    # pprint(results)
    print("Search results:")
    for result in results:
        print(f"- {result.title} ({result.page_id})")
        print(f" Summary: {result.snippet}")
        print("=" * 20)
    print(f"Suggestion: {results.suggestion}")

    # Get page content
    # page = await client.get_page(page_id=21356332)
    # print(f"Title: {page.title}")
    # print(f"Summary: {page.summary[:200]}...")

    # # Batch operations
    # pages = await client.get_pages_batch(["Python", "JavaScript", "Rust"])
    # for p in pages:
    #     print(f"Batch page: {p.title} - {p.summary[:100]}...")
    # # Close the client session

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
