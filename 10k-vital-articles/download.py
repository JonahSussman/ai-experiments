import json
import time
import requests
import itertools
from pathlib import Path

EXCLUDE_BATCHES = [0]

petscan: dict = json.load(open('../10k-vital-articles/petscan.json'))

# # Sort based on pageid
# petscan["*"][0]["a"]["*"].sort(key=lambda x: x["id"])
# with open("data/petscan.json", "w") as f:
#     json.dump(petscan, f)

# TODO: Continues!

Path("data/chunks").mkdir(exist_ok=True)

base_query = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&redirects=1&formatversion=2&rvprop=content&rvslots=main&rvsection=intro&rvcontentformat-main=text%2Fx-wiki&pageids="

for i, batch in enumerate(itertools.batched(petscan["*"][0]["a"]["*"], 50)):
    if i in EXCLUDE_BATCHES:
        continue

    query = base_query + "|".join(str(x["id"]) for x in batch)

    print(f"[{i}/{len(petscan["*"][0]["a"]["*"]) // 50}] pageid from {batch[0]['id']} to {batch[-1]['id']} ({batch[0]["title"]} to {batch[-1]["title"]})")
    print("    " + query)

    response = requests.get(query)

    with open(f"data/chunks/{i}.json", "w") as f:
        try:
            data = response.json()
            json.dump(data, f)

            if "batchcomplete" not in data or bool(data["batchcomplete"]) is False:
                exit(1)
            
        except Exception as e:
            f.write(f"{response}, {e}")
            exit(1)

    time.sleep(0.5)  # Be nice to the API
