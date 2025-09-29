from wikipedia_async.helpers.section_helpers import SectionHelper
import asyncio
import time
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# section = None
# page = None


# async def main():
#     global section, page
#     # Initialize client with optimal defaults
#     client = WikipediaClient(
#         ClientConfig(
#             cache_type="file",
#             cache_ttl=3600 * 24,
#             cache_serializer="json",
#             # enable_cache=False,
#         )
#     )
#     # s = time.time()
#     # await client.set_language("bn")
#     titles = [
#         "Python (programming language)",
#         "Java (programming language)",
#         "C (programming language)",
#     ]

#     s = time.time()
#     title = titles[0]
#     page = await client.get_page(title, lang="en")
#     # print("Time taken:", time.time() - s)

#     # print(page.helper.to_json(keep_links=False, content_limit=500))
#     print(page.helper.to_string(markdown=True, keep_links=False))

#     await client.close()

#     # pprint(page.helper.tree_view_json(50))

#     # pprint(
#     #     page.helper.to_json(
#     #         keep_links=False,
#     #         # table_limit=1,
#     #         content_limit=500,
#     #         show_children=False,
#     #     )
#     # )


# if __name__ == "__main__":
#     asyncio.run(main())

with open(
    r"D:\code\python\test\wiki_search_test\wiki_cache\cache_3a2826accf04b6d0185150ec01bae9a8.json"
) as f:
    html = json.load(f)["query"]["pages"][0]["revisions"][0]["content"]

with open("test.html", "w", encoding="utf-8") as f:
    f.write(html)

helper = SectionHelper.from_html(html, "en")


# print(helper.to_string(markdown=True, keep_links=False))
