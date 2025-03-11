import json

from typing import Optional
from wikitexthtml import Page
from pydantic import BaseModel

class PageInfo(BaseModel):
    title: str
    page_id: int
    wiki_intro: str

PAGES_RAW = json.load(open("data/pages.json"))
PAGES = [PageInfo(**page) for page in PAGES_RAW]

class ThePage(Page):
    def __init__(self, info: PageInfo):
        self.info = info

        super().__init__(info.title)

    def page_load(self, page: str) -> str:
        return self.info.wiki_intro
    
    def page_exists(self, page):
        return True
    
    def template_exists(self, template: str) -> bool:
        print(f"Checking if template `{template}` exists")
        return False
    
    def template_load(self, template: str) -> str:
        print(f"Loading template `{template}`")
        return ""
    
    def clean_title(self, title):
        print(f"Cleaning title `{title}`")
        return self.info.title
    
    def clean_url(self, url):
        print(f"Cleaning URL `{url}`")
        return url




print(ThePage(PAGES[0]).render().html)