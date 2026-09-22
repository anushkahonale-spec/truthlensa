import joblib
import os

from services.web_verifier import verify_claim
from services.gemini_verifier import verify_news_claim


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    "models",
    "text_classifier.pkl"
)

VECTORIZER_PATH = os.path.join(
    "models",
    "text_vectorizer.pkl"
)

model = joblib.load(MODEL_PATH)

vectorizer = joblib.load(
    VECTORIZER_PATH
)


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
        word
        for word in SENSATIONAL_WORDS
        if word in text_lower
    ]


    urgency_found = [
        word
        for word in URGENCY_WORDS
        if word in text_lower
    ]


    clickbait_found = [
        phrase
        for phrase in CLICKBAIT_PHRASES
        if phrase in text_lower
    ]


    if sensational_found:

        signals.append(
            "Sensational language detected: "
            + ", ".join(
                sensational_found
            )
        )


    if urgency_found:

        signals.append(
            "Urgency or sharing pressure detected: "
            + ", ".join(
                urgency_found
            )
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
# MAIN ANALYSIS
# =========================================================

def analyze_text(text):

    # -----------------------------------------------------
    # EMPTY INPUT
    # -----------------------------------------------------

    if not text or not text.strip():

        return {

            "status": "uncertain",

            "confidence": 0,

            "ml_confidence": 0,

            "message": (
                "Please enter some content to analyze."
            ),

            "signals": [
                "No content was provided."
            ],

            "verification": {

                "status": "uncertain",

                "verdict": "UNCERTAIN",

                "confidence": 0,

                "truth_score": 50,

                "analysis": (
                    "No claim was provided."
                ),

                "sources": []

            }

        }


    text = text.strip()


    # -----------------------------------------------------
    # LOCAL ML MODEL
    # -----------------------------------------------------

    text_features = vectorizer.transform(
        [text]
    )

    probabilities = model.predict_proba(
        text_features
    )[0]

    ml_confidence = float(
        max(probabilities) * 100
    )


    # -----------------------------------------------------
    # LANGUAGE SIGNALS
    # -----------------------------------------------------

    language_signals = analyze_language_patterns(
        text
    )


    # -----------------------------------------------------
    # GEMINI VERIFICATION
    # -----------------------------------------------------

    gemini_result = verify_news_claim(
        text
    )


    # -----------------------------------------------------
    # NEWSAPI FALLBACK
    # -----------------------------------------------------

    news_result = None

    if (
        gemini_result.get("status")
        == "verification_unavailable"
    ):

        news_result = verify_claim(
            text
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    if (
        gemini_result.get("status")
        == "supported"
    ):

        status = "supported"


        # IMPORTANT:
        # Display factual support score,
        # NOT Gemini confidence.

        confidence = gemini_result.get(
            "truth_score",
            100
        )


        message = (
            "Gemini found reliable evidence "
            "supporting this claim."
        )


        verification = gemini_result


    elif (
        gemini_result.get("status")
        == "conflicting"
    ):

        status = "conflicting"


        # IMPORTANT:
        # A false claim should NOT display
        # Gemini's confidence as 100%.
        #
        # Display the factual support score.

        confidence = gemini_result.get(
            "truth_score",
            0
        )


        message = (
            "Gemini found reliable evidence "
            "that conflicts with this claim."
        )


        verification = gemini_result


    elif (
        gemini_result.get("status")
        == "uncertain"
    ):

        status = "uncertain"


        # For uncertain claims, use Gemini's
        # estimated factual support score.

        confidence = gemini_result.get(
            "truth_score",
            50
        )


        message = (
            "Available real-time evidence was not "
            "sufficient to establish the claim."
        )


        verification = gemini_result


    else:

        # -------------------------------------------------
        # NEWSAPI FALLBACK
        # -------------------------------------------------

        if news_result:

            verification = news_result

            status = "uncertain"

            confidence = 0

            message = (
                "Gemini verification was unavailable. "
                "News sources are shown only as supporting "
                "context and are not treated as proof."
            )

        else:

            verification = gemini_result

            status = "uncertain"

            confidence = 0

            message = (
                "Real-time verification is temporarily "
                "unavailable."
            )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {

        "status": status,

        # This is now the TRUTH SCORE shown in the UI.
        "confidence": confidence,

        # Original local ML confidence remains available.
        "ml_confidence": round(
            ml_confidence,
            2
        ),

        "message": message,

        "signals": language_signals,

        "verification": verification

    }