import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than",
    "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "from",
    "with", "by", "at", "as", "it", "its", "into", "about", "after",
    "before", "during", "over", "under", "has", "have", "had",
    "will", "would", "can", "could", "may", "might", "should",
    "do", "does", "did", "not", "no", "news", "report", "reports"
}


def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_keywords(text):
    cleaned = clean_text(text)
    words = cleaned.split()

    keywords = []

    for word in words:
        if len(word) < 3:
            continue

        if word in STOP_WORDS:
            continue

        if word not in keywords:
            keywords.append(word)

    return keywords[:10]


def build_search_query(text):
    keywords = extract_keywords(text)

    if not keywords:
        return ""

    # For very short claims, searching every word together
    # can return poor results. Use the most meaningful words.
    return " AND ".join(keywords[:6])


def normalize_for_comparison(text):
    return set(extract_keywords(text))


def keyword_overlap(claim, article_text):
    claim_words = normalize_for_comparison(claim)
    article_words = normalize_for_comparison(article_text)

    if not claim_words:
        return 0.0

    common = claim_words.intersection(article_words)

    return len(common) / len(claim_words)


def fetch_news(query):
    if not NEWS_API_KEY:
        return {
            "status": "error",
            "message": "NEWS_API_KEY is missing.",
            "articles": []
        }

    try:
        response = requests.get(
            NEWS_API_URL,
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": NEWS_API_KEY
            },
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return {
            "status": "success",
            "message": "",
            "articles": data.get("articles", [])
        }

    except requests.RequestException as exc:
        return {
            "status": "error",
            "message": str(exc),
            "articles": []
        }


def build_source(article, claim):
    title = article.get("title") or ""
    description = article.get("description") or ""
    content = article.get("content") or ""

    combined_text = f"{title} {description} {content}"

    overlap = keyword_overlap(claim, combined_text)

    return {
        "title": title,
        "description": description,
        "url": article.get("url"),
        "source": (
            article.get("source", {}).get("name")
            if article.get("source")
            else None
        ),
        "published_at": article.get("publishedAt"),
        "relevance": round(overlap * 100, 2)
    }


def verify_claim(text):
    if not text or not text.strip():
        return {
            "status": "uncertain",
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "analysis": "No claim was provided.",
            "sources": []
        }

    query = build_search_query(text)

    if not query:
        return {
            "status": "uncertain",
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "analysis": "The claim did not contain enough meaningful keywords.",
            "sources": []
        }

    result = fetch_news(query)

    if result["status"] != "success":
        return {
            "status": "error",
            "verdict": "verification_unavailable",
            "confidence": 0,
            "analysis": result["message"],
            "sources": []
        }

    articles = result["articles"]

    if not articles:
        return {
            "status": "no_sources",
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "analysis": (
                "No matching recent news articles were found "
                "for the submitted claim."
            ),
            "sources": []
        }

    sources = [
        build_source(article, text)
        for article in articles
    ]

    # Sort strongest matches first.
    sources.sort(
        key=lambda item: item["relevance"],
        reverse=True
    )

    # IMPORTANT:
    # Do not treat weak keyword matches as evidence.
    relevant_sources = [
        source
        for source in sources
        if source["relevance"] >= 50
    ]

    # Keep only genuinely relevant sources.
    relevant_sources = relevant_sources[:5]

    if not relevant_sources:
        return {
            "status": "no_sources",
            "verdict": "insufficient_evidence",
            "confidence": 0,
            "analysis": (
                "Recent articles were found, but none matched "
                "the claim strongly enough to be considered "
                "relevant evidence."
            ),
            "sources": []
        }

    return {
        "status": "sources_found",
        "verdict": "evidence_found",
        "confidence": min(
            50 + len(relevant_sources) * 5,
            75
        ),
        "analysis": (
            "Relevant news sources were found. "
            "These sources provide context but do not "
            "by themselves establish whether the claim "
            "is true or false."
        ),
        "sources": relevant_sources
    }

