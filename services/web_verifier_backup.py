import os
import re
import requests

from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def build_search_query(text):
    """Create a concise search query from the submitted claim."""

    stop_words = {
        "the", "a", "an", "and", "or", "but", "for", "to", "of",
        "in", "on", "at", "by", "with", "from", "is", "was", "were",
        "has", "have", "had", "this", "that", "these", "those", "it",
        "its", "as", "be", "been", "are", "will", "would", "can",
        "could", "should", "may", "might", "their", "they", "them",
        "who", "which", "what", "when", "where", "how"
    }

    words = re.findall(r"[A-Za-z0-9₹]+", text.strip())

    useful_words = [
        word for word in words
        if word.lower() not in stop_words and len(word) > 2
    ]

    return " ".join(useful_words[:10])


def verify_claim(text):
    """
    NewsAPI fallback.

    Important:
    Finding related articles does NOT prove that the submitted
    claim is true. This function only retrieves recent evidence
    that can be shown to the user.
    """

    if not text or not text.strip():
        return {
            "status": "uncertain",
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "analysis": "No claim was provided for verification.",
            "sources": []
        }

    if not NEWS_API_KEY:
        return {
            "status": "error",
            "verdict": "verification_unavailable",
            "confidence": 0,
            "analysis": "NewsAPI key is not configured.",
            "sources": []
        }

    search_query = build_search_query(text)

    if not search_query:
        return {
            "status": "uncertain",
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "analysis": "A useful search query could not be created.",
            "sources": []
        }

    params = {
        "q": search_query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    try:
        response = requests.get(
            NEWS_API_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if data.get("status") != "ok":
            return {
                "status": "error",
                "verdict": "verification_unavailable",
                "confidence": 0,
                "analysis": data.get(
                    "message",
                    "News verification could not be completed."
                ),
                "sources": []
            }

        articles = data.get("articles", [])
        sources = []

        for article in articles:
            title = article.get("title", "")

            if not title:
                continue

            sources.append({
                "title": title,
                "url": article.get("url", ""),
                "source": (
                    article.get("source", {})
                    .get("name", "Unknown source")
                ),
                "published_at": article.get(
                    "publishedAt", ""
                ),
                "description": (
                    article.get("description")
                    or "No description available."
                )
            })

        if not sources:
            return {
                "status": "no_sources",
                "verdict": "insufficient_evidence",
                "confidence": 0,
                "analysis": (
                    "No recent articles matching the main "
                    "keywords of this claim were found."
                ),
                "search_query": search_query,
                "sources": []
            }

        return {
            "status": "sources_found",
            "verdict": "uncertain",
            "confidence": 0,
            "analysis": (
                f"Found {len(sources)} recent articles related "
                "to the main keywords in this claim. These "
                "sources provide context but do not by themselves "
                "prove that the claim is true or false."
            ),
            "search_query": search_query,
            "sources": sources
        }

    except requests.RequestException:
        return {
            "status": "error",
            "verdict": "verification_unavailable",
            "confidence": 0,
            "analysis": (
                "The NewsAPI verification service could not "
                "be reached."
            ),
            "sources": []
        }

    except Exception as error:
        return {
            "status": "error",
            "verdict": "verification_unavailable",
            "confidence": 0,
            "analysis": (
                "An unexpected error occurred during "
                "news verification."
            ),
            "sources": [],
            "error": str(error)
        }