import json
import time
import requests
import itertools

petscan: dict = json.load(open('../10k-vital-articles/petscan.json'))

# # Sort based on pageid
# petscan["*"][0]["a"]["*"].sort(key=lambda x: x["id"])
# with open("data/petscan.json", "w") as f:
#     json.dump(petscan, f)

# TODO: Continues!

base_query = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&pageids="

for i, batch in enumerate(itertools.batched(petscan["*"][0]["a"]["*"], 50)):
    query = base_query + "|".join(str(x["id"]) for x in batch)

    print(f"[{i}/{len(petscan["*"][0]["a"]["*"]) // 50}] pageid from {batch[0]['id']} to {batch[-1]['id']} ({batch[0]["title"]} to {batch[-1]["title"]})")
    print("    " + query)

    response = requests.get(query)

    with open(f"data/{i}.json", "w") as f:
        try:
            data = response.json()
            json.dump(data, f)
        except Exception as e:
            f.write(f"{response}, {e}")

    time.sleep(0.5)  # Be nice to the API
