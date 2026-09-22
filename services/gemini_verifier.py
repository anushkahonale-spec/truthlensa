import os
import re

from dotenv import load_dotenv
from google import genai


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# =========================================================
# GEMINI VERIFICATION
# =========================================================

def verify_news_claim(text):

    # -----------------------------------------------------
    # EMPTY INPUT
    # -----------------------------------------------------

    if not text or not text.strip():

        return {
            "status": "uncertain",
            "verdict": "UNCERTAIN",
            "confidence": 0,
            "truth_score": 50,
            "analysis": "No claim was provided.",
            "sources": []
        }


    # -----------------------------------------------------
    # GEMINI API CHECK
    # -----------------------------------------------------

    if client is None:

        return {
            "status": "verification_unavailable",
            "verdict": "UNCERTAIN",
            "confidence": 0,
            "truth_score": 50,
            "analysis": "Gemini API key is not configured.",
            "sources": []
        }


    # =====================================================
    # GEMINI PROMPT
    # =====================================================

    prompt = f"""
You are the fact verification engine for TruthLens.

Analyze the following submitted claim carefully:

--- CLAIM START ---
{text}
--- CLAIM END ---

Determine whether the claim is:

SUPPORTED
CONFLICTING
UNCERTAIN

Your task is to evaluate the FACTUAL SUPPORT contained in the
submitted claim.

IMPORTANT TRUTH SCORE RULES:

TRUTH_SCORE is NOT your confidence.

TRUTH_SCORE measures how much of the actual claim is factually
supported.

Use this scale:

100 = The complete claim is factually supported.
90-99 = Almost completely supported, with only a very minor issue.
70-89 = Mostly supported, but contains a meaningful minor error
       or unsupported detail.
50-69 = Partially supported and partially incorrect or unsupported.
30-49 = Mostly incorrect, but contains some factual element.
10-29 = Almost completely false, with only a small factual element.
0-9 = Completely false or directly contradicted by reliable facts.
0 = The complete claim is false or directly contradicted.

For a simple completely true factual statement, use close to 100.

For a simple completely false factual statement, use close to 0.

For a mixed claim containing both true and false information,
calculate a reasonable score between 1 and 99 based on how much
of the claim is actually supported.

Do NOT automatically give 100 simply because you are highly
confident in your answer.

CONFIDENCE means how confident you are that your VERDICT and
TRUTH_SCORE assessment is correct.

VERDICT rules:

SUPPORTED:
Use when the main claim is factually supported.

CONFLICTING:
Use when the main claim is factually false or contradicted.

UNCERTAIN:
Use only when reliable information is insufficient or the claim
cannot reasonably be determined.

For established scientific facts, use established scientific
knowledge.

For current events, be careful about the possibility that
information may have changed.

Do not invent sources.

For political or election claims, remain neutral and factual.

Keep ANALYSIS short but explain the main reason for the verdict.

Return ONLY these five lines:

VERDICT: SUPPORTED
TRUTH_SCORE: 100
CONFIDENCE: 100
ANALYSIS: Short explanation.
SOURCES: None
"""


    # =====================================================
    # CALL GEMINI
    # =====================================================

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        output = response.text or ""

        print("Gemini response:")
        print(output)


        # =================================================
        # VERDICT
        # =================================================

        verdict_match = re.search(
            r"VERDICT\s*:\s*(SUPPORTED|CONFLICTING|UNCERTAIN)",
            output,
            re.IGNORECASE
        )

        verdict = (
            verdict_match.group(1).upper()
            if verdict_match
            else "UNCERTAIN"
        )


        # =================================================
        # TRUTH SCORE
        # =================================================

        truth_score_match = re.search(
            r"TRUTH_SCORE\s*:\s*(\d{1,3})",
            output,
            re.IGNORECASE
        )

        if truth_score_match:

            truth_score = int(
                truth_score_match.group(1)
            )

            truth_score = max(
                0,
                min(100, truth_score)
            )

        else:

            # Safe fallback based on verdict
            if verdict == "SUPPORTED":

                truth_score = 100

            elif verdict == "CONFLICTING":

                truth_score = 0

            else:

                truth_score = 50


        # =================================================
        # CONFIDENCE
        # =================================================

        confidence_match = re.search(
            r"CONFIDENCE\s*:\s*(\d{1,3})",
            output,
            re.IGNORECASE
        )

        if confidence_match:

            confidence = int(
                confidence_match.group(1)
            )

            confidence = max(
                0,
                min(100, confidence)
            )

        else:

            confidence = 0


        # =================================================
        # ANALYSIS
        # =================================================

        analysis_match = re.search(
            r"ANALYSIS\s*:\s*(.*?)(?:\nSOURCES\s*:|\Z)",
            output,
            re.IGNORECASE | re.DOTALL
        )

        analysis = (
            analysis_match.group(1).strip()
            if analysis_match
            else "Gemini analyzed the submitted claim."
        )


        # =================================================
        # STATUS
        # =================================================

        if verdict == "SUPPORTED":

            status = "supported"

        elif verdict == "CONFLICTING":

            status = "conflicting"

        else:

            status = "uncertain"


        # =================================================
        # FINAL RESULT
        # =================================================

        return {

            "status": status,

            "verdict": verdict,

            # Confidence in Gemini's assessment
            "confidence": confidence,

            # Factual support score
            "truth_score": truth_score,

            "analysis": analysis,

            "sources": []

        }


    # =====================================================
    # GEMINI ERROR
    # =====================================================

    except Exception as exc:

        print(
            "Gemini verification error:",
            exc
        )

        return {

            "status": "verification_unavailable",

            "verdict": "UNCERTAIN",

            "confidence": 0,

            "truth_score": 50,

            "analysis": (
                "Gemini verification failed: "
                + str(exc)
            ),

            "sources": []

        }

