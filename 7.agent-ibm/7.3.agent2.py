from crewai.tools import tool
from bs4 import BeautifulSoup
import requests

@tool("Free DuckDuckGo Search")
def free_ddg_search(query: str) -> str:
  """Performs a free web search using DuckDuckGo Lite HTML without API keys."""
  url = "https://lite.duckduckgo.com/lite/"
  headers = {"User-Agent": "Mozilla/5.0"}
  data = {"q": query}

  response = requests.post(url, data=data, headers=headers)
  if response.status_code != 200:
    return "Search failed."

  soup = BeautifulSoup(response.text, "html.parser")
  results = []
  for tr in soup.find_all("tr"):
    snippets = tr.find_all("td", class_="result-snippet")
    if snippets:
      results.append(snippets[0].get_text(strip=True))

  return "\n".join(results[:5]) if results else "No results found."
