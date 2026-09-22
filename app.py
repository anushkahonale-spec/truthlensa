from flask import Flask, render_template, request, jsonify

from services.text_analyzer import analyze_text


app = Flask(__name__)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Text Checker page
@app.route("/text-checker")
def text_checker():
    return render_template("text-checker.html")


# Text analysis API
@app.route("/analyze-text", methods=["POST"])
def analyze_text_route():

    data = request.get_json()

    text = data.get("text", "").strip()

    result = analyze_text(text)

    return jsonify(result)


# Start Flask
if __name__ == "__main__":
    app.run(debug=True)