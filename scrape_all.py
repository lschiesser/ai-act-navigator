from article_scraper import AIActScraper
import json
from tqdm import tqdm
import time

articles = []
for i in range(1, 114): #43, 44):
    articles.append(("https://artificialintelligenceact.eu/article/" + str(i) + "/", i))

annexes = []
for i in range(1, 14):
    annexes.append(("https://artificialintelligenceact.eu/annex/" + str(i) + "/", i))

recitals = []
for i in range(1, 181):
    recitals.append(("https://artificialintelligenceact.eu/recital/" + str(i) + "/", i))

json_kwargs = {'ensure_ascii': False, 'indent': 2, 'sort_keys': True}

scraper = AIActScraper()

for article in tqdm(articles, desc="Scraping articles"):
    result = scraper.scrape_from_url(article[0])

    with open(f"./data/articles/article_{article[1]}.json", "w") as f:
        json.dump(result, f, **json_kwargs)
    
    time.sleep(1)

for annex in tqdm(annexes, desc="Scraping annexes"):
    result = scraper.scrape_from_url(annex[0])

    with open(f"./data/annexes/annex_{annex[1]}.json", "w") as f:
        json.dump(result, f, **json_kwargs)
    
    time.sleep(1)

for recital in tqdm(recitals, desc="Scraping recitals"):
    result = scraper.scrape_from_url(recital[0])
    with open(f"./data/recitals/recital_{recital[1]}.json", "w") as f:
        json.dump(result, f, **json_kwargs)
    time.sleep(1)
