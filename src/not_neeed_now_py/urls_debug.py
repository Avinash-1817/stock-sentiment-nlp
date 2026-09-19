import requests
from bs4 import BeautifulSoup

url = "https://www.moneycontrol.com/news/business/economy/covid-19-pandemic-us-consumer-spending-tumble-in-april-5334861.html"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers, timeout=10)
print("Status code:", resp.status_code)

soup = BeautifulSoup(resp.content, "html.parser")

# print all meta tags to see what's actually there
for tag in soup.find_all("meta"):
    if tag.get("property") or tag.get("name"):
        key = tag.get("property") or tag.get("name")
        if "date" in key.lower() or "time" in key.lower() or "publish" in key.lower():
            print(key, "->", tag.get("content"))

# check for JSON-LD blocks
import json
for script in soup.find_all("script", type="application/ld+json"):
    try:
        data = json.loads(script.string)
        print(json.dumps(data, indent=2)[:1000])
    except Exception as e:
        print("JSON-LD parse error:", e)