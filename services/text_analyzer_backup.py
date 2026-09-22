
import joblib
import os

from services.web_verifier import verify_claim


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join("models", "text_classifier.pkl")
VECTORIZER_PATH = os.path.join("models", "text_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =========================================================
# LANGUAGE PATTERN CHECKS
# =========================================================

SENSATIONAL_WORDS = [
    "shocking",
    "unbelievable",
    "secret",
    "miracle",
    "exposed",
    "explosive",
    "stunning",
    "scandal",
    "you won't believe",
    "you will not believe"
]

URGENCY_WORDS = [
    "breaking",
    "urgent",
    "immediately",
    "share now",
    "share this",
    "before it gets deleted",
    "act now"
]

CLICKBAIT_PHRASES = [
    "you won't believe",
    "you will never believe",
    "what happens next",
    "they don't want you to know",
    "click here",
    "share before"
]


def analyze_language_patterns(text):

    text_lower = text.lower()

    signals = []

    sensational_found = [
        word for word in SENSATIONAL_WORDS
        if word in text_lower
    ]

    urgency_found = [
        word for word in URGENCY_WORDS
        if word in text_lower
    ]

    clickbait_found = [
        phrase for phrase in CLICKBAIT_PHRASES
        if phrase in text_lower
    ]

    if sensational_found:
        signals.append(
            "Sensational language detected: "
            + ", ".join(sensational_found)
        )

    if urgency_found:
        signals.append(
            "Urgency or sharing pressure detected: "
            + ", ".join(urgency_found)
        )

    if clickbait_found:
        signals.append(
            "Possible clickbait phrasing detected."
        )

    if not signals:
        signals.append(
            "No obvious sensational or clickbait language detected."
        )

    return signals


# =========================================================
# MAIN TEXT ANALYSIS
# =========================================================

def analyze_text(text):

    # -----------------------------------------------------
    # EMPTY INPUT
    # -----------------------------------------------------

    if not text or not text.strip():

        return {
            "status": "uncertain",
            "confidence": 0,
            "message": "Please enter some content to analyze.",
            "signals": [
                "No content was provided."
            ],
            "verification": {
                "status": "uncertain",
                "verdict": "insufficient_evidence",
                "confidence": 0,
                "analysis": "No claim was provided for verification.",
                "sources": []
            }
        }

    text = text.strip()


    # -----------------------------------------------------
    # LOCAL LANGUAGE MODEL
    # -----------------------------------------------------

    text_features = vectorizer.transform([text])

    probabilities = model.predict_proba(text_features)[0]

    ml_confidence = float(max(probabilities) * 100)


    # -----------------------------------------------------
    # LANGUAGE SIGNALS
    # -----------------------------------------------------

    language_signals = analyze_language_patterns(text)


    # -----------------------------------------------------
    # EXTERNAL VERIFICATION
    #
    # Gemini is temporarily skipped because its Google
    # Search quota is exhausted.
    #
    # NewsAPI is used directly as a fallback.
    # -----------------------------------------------------

    verification = verify_claim(text)


    # -----------------------------------------------------
    # FINAL STATUS
    #
    # NewsAPI finding related articles does NOT prove
    # that the submitted claim is true.
    # Therefore the result remains UNCERTAIN.
    # -----------------------------------------------------

    status = "uncertain"

    message = (
        "The claim could not be fact-checked with "
        "sufficient external evidence."
    )


    if verification.get("status") == "sources_found":

        status = "uncertain"

        message = (
            "Related news sources were found, but "
            "their presence alone does not prove "
            "that the submitted claim is true or false."
        )


    elif verification.get("status") == "no_sources":

        status = "uncertain"

        message = (
            "No matching recent news sources were "
            "found, so the claim remains uncertain."
        )


    elif verification.get("status") in {
        "error",
        "verification_unavailable"
    }:

        status = "uncertain"

        message = (
            "External verification is temporarily "
            "unavailable. Language-pattern signals "
            "are shown separately."
        )


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return {

        "status": status,

        # IMPORTANT:
        # This is language-model confidence only.
        # It is NOT probability that the claim is true.
        "confidence": round(
            ml_confidence,
            2
        ),

        "message": message,

        "signals": language_signals,

        "verification": verification

    }

