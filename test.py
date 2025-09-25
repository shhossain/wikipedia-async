from wikipedia_async import WikipediaClient, ClientConfig, SectionHelper
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

    s = time.time()
    res = await client.get_pages_batch(titles, lang="en")
    print("Time taken:", time.time() - s)
    for page in res.successful:
        print("Title:", page.title)
        print("Summary:", page.summary)
        print(page.helper.tree_view(100))
        print("=" * 80)
    print("Successful:", len(res.successful))
    print("Failed:", len(res.failed))

    for fail in res.failed:
        print(f"Failed to get {fail['title']}: {fail['error']}")

    for page in res.successful:
        print("Title:", page.title)
        print("Summary:", page.summary)
        print(page.helper.tree_view(100))
        print("=" * 80)
        for sec in page.sections:
            print(sec.title)
            print(sec.to_string(markdown=True)[:200])

    # print(list(page.tables.items())[3])

    # sums = await client.get_summary_batch(titles)
    # for title, summ in zip(titles, sums):
    #     print(f"Title: {title}")
    #     print(summ)
    #     print("=" * 80)

    await client.close()


if __name__ == "__main__":

    asyncio.run(main())
