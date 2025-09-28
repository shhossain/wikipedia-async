from pprint import pprint
from wikipedia_async import WikipediaClient, ClientConfig
import asyncio
import time

section = None


async def main():
    global section
    # Initialize client with optimal defaults
    client = WikipediaClient(
        ClientConfig(
            cache_type="file",
            cache_ttl=3600 * 24,
            cache_serializer="json",
            # enable_cache=False,
        )
    )
    # s = time.time()
    # await client.set_language("bn")
    titles = [
        "List of constituencies of the Jatiya Sangsad",
        "Java (programming language)",
        "C (programming language)",
    ]

    s = time.time()
    title = titles[0]
    page = await client.get_page(title, lang="en")
    print("Time taken:", time.time() - s)
    for sec in page.sections:
        for table in sec.tables:
            print(table.caption)
            print(table.dataframe)
            print()
    
    await client.close()

    # pprint(page.helper.tree_view_json(50))

    # pprint(
    #     page.helper.to_json(
    #         keep_links=False,
    #         # table_limit=1,
    #         content_limit=500,
    #         show_children=False,
    #     )
    # )

    


# if __name__ == "__main__":

asyncio.run(main())
