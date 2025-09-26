from pprint import pprint
from wikipedia_async import WikipediaClient, ClientConfig
import asyncio
import time


async def main():
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
        "Python (programming language)",
        "Java (programming language)",
        "C (programming language)",
    ]

    # s = time.time()
    # res = await client.get_page_html_batch(titles)
    # for title, html in zip(titles, [r.html for r in res.successful]):
    #     print(f"Title: {title}")
    #     print(html[:200])
    #     print("=" * 80)
    # print("Time taken:", time.time() - s)
    # print("Successful:", len(res.successful))
    # print("Failed:", len(res.failed))

    # for title in titles:
    #     page = await client.get_page_html(title)
    #     print(f"Title: {title}")
    #     print(page[:200])
    #     print("=" * 80)
    # print("Time taken:", time.time() - s)

    # s = time.time()
    # res = await client.get_pages_batch(titles, lang="en")
    # print("Time taken:", time.time() - s)
    # for page in res.successful:
    #     print("Title:", page.title)
    #     print("Summary:", page.summary)
    #     print(page.helper.tree_view(100))
    #     print("=" * 80)
    # print("Successful:", len(res.successful))
    # print("Failed:", len(res.failed))

    # for fail in res.failed:
    #     print(f"Failed to get {fail['title']}: {fail['error']}")

    # for page in res.successful:
    #     print("Title:", page.title)
    #     print("Summary:", page.summary)
    #     print(page.helper.tree_view())
    #     print("=" * 80)
    #     for sec in page.sections:
    #         print(sec.to_string(markdown=False)[:200])

    # print(list(page.tables.items())[3])

    # sums = await client.get_summary_batch(titles)
    # for title, summ in zip(titles, sums):
    #     print(f"Title: {title}")
    #     print(summ)
    #     print("=" * 80)

    s = time.time()
    page = await client.get_page(titles[0], lang="en")
    print("Time taken:", time.time() - s)

    # pprint(page.helper.tree_view_json(50))

    # pprint(
    #     page.helper.to_json(
    #         keep_links=False,
    #         table_limit=1,
    #         content_limit=500,
    #         show_children=False,
    #     )
    # )

    pprint(
        page.sections[0].to_json(
            keep_links=False,
            content_start_index=100,
            content_limit=400,
        )
    )
    print("=" * 20)
    pprint(page.sections[0].tree_view_json())

    await client.close()


if __name__ == "__main__":

    asyncio.run(main())
