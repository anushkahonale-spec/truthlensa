from flask import Flask, render_template, request, jsonify

from services.text_analyzer import analyze_text
from services.image_analyzer import analyze_image


app = Flask(__name__)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Text Checker page
@app.route("/text-checker")
def text_checker():
    return render_template("text-checker.html")


# Image Checker page
@app.route("/image-checker")
def image_checker():
    return render_template("image-checker.html")


# Text analysis API
@app.route("/analyze-text", methods=["POST"])
def analyze_text_route():

    data = request.get_json()

    text = data.get("text", "").strip()

    result = analyze_text(text)

    return jsonify(result)


# Image analysis API
@app.route("/analyze-image", methods=["POST"])
def analyze_image_route():

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image file uploaded."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No image selected."
        }), 400

    try:
        image_bytes = image_file.read()

        result = analyze_image(image_bytes)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Unable to process image: {str(e)}"
        }), 500


# Start Flask
if __name__ == "__main__":
    app.run(debug=True)