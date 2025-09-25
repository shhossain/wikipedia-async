from wikipedia_async import WikipediaClient, ClientConfig, SectionHelper
import asyncio
async def main():
    # Initialize client with optimal defaults
    client = WikipediaClient(
        ClientConfig(
            cache_type="file",
            cache_ttl=3600 * 24,
            cache_serializer="json",
        )
    )
    # s = time.time()
    # await client.set_language("bn")

    page = await client.get_page(
        "Python (programming language)"
    )
    print("Summary:", page.summary)
    print(page.helper.tree_view(100))
    # for sec in page.sections:
    #     print(sec.title)
    #     print(sec.to_string(markdown=True)[:200])

    # print(list(page.tables.items())[3])

    await client.close()


if __name__ == "__main__":

    asyncio.run(main())
