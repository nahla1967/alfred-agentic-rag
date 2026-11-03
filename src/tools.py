import requests
from ddgs import DDGS
from huggingface_hub import list_models
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the public web for general information."""
    results = DDGS().text(query, max_results=5)
    return "\n".join(f"{r['title']}: {r['body']}" for r in results)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo.raise_for_status()
    data = geo.json()

    if not data.get("results"):
        return f"City not found: {city}"

    latitude = data["results"][0]["latitude"]
    longitude = data["results"][0]["longitude"]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        },
        timeout=10,
    )
    weather.raise_for_status()
    current = weather.json()["current"]

    return (
        f"Weather in {city}: "
        f"Temperature: {current['temperature_2m']}°C, "
        f"Humidity: {current['relative_humidity_2m']}%, "
        f"Wind speed: {current['wind_speed_10m']} km/h"
    )


@tool
def huggingface_search(task: str) -> str:
    """Search Hugging Face models for a specific machine learning task
    (for example: text-classification, summarization)."""
    models = list_models(pipeline_tag=task, limit=5)
    return "\n".join(model.id for model in models)
@tool
def hub_stats(author: str) -> str:
    """Find the most downloaded Hugging Face model by an author."""
    models = list_models(
        author=author,
        sort="downloads",
        limit=1
)

    models = list(models)

    if not models:
        return f"No models found for {author}"

    model = models[0]

    return (
        f"The most downloaded model by {author} "
        f"is {model.id} with {model.downloads:,} downloads."
    )

if __name__ == "__main__":
    print("Web search:")
    print(web_search.invoke({"query": "Latest artificial intelligence news"}))

    print("\nWeather:")
    print(get_weather.invoke({"city": "Tunis"}))

    print("\nHugging Face:")
    print(huggingface_search.invoke({"task": "text-classification"}))