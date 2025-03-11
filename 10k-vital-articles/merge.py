import json
from pydantic import BaseModel
from pathlib import Path

class PageInfo(BaseModel):
    title: str
    page_id: int
    wiki_intro: str

pages: list[PageInfo] = []

# for every file in the data/ dir
for file_path in Path("data/chunks").iterdir():
    print(file_path)
    with open(file_path) as f:
        data = json.load(f)

        for page in data["query"]["pages"]:
            title = page["title"]
            page_id = page["pageid"]
            wiki_intro = page["revisions"][0]["slots"]["main"]["content"]
            pages.append(PageInfo(title=title, page_id=page_id, wiki_intro=wiki_intro))

json.dump([x.model_dump() for x in pages], open("data/pages.json", "w"), indent=2)