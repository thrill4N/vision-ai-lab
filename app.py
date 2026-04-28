from flask import Flask, render_template, request, jsonify
import requests, os, json
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from openai import OpenAI

load_dotenv()

# API CONFIG
HF_API_URL = os.getenv("HUGGING_FACE_API_URL")
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# 🔹 Query Hugging Face model
def query_model(image_bytes):
    response = requests.post(HF_API_URL, headers=HF_HEADERS, data=image_bytes)
    return json.loads(response.content.decode("utf-8"))

# 🔹 Extract dominant color
def get_dominant_color(file):
    image = Image.open(file).convert("RGB").resize((100, 100))
    pixels = np.array(image).reshape(-1, 3)
    avg_color = pixels.mean(axis=0).astype(int)
    return f"RGB({avg_color[0]}, {avg_color[1]}, {avg_color[2]})"

# 🔹 Ask LLM
def ask_llm(predictions, color, question):
    labels = ", ".join([p["label"] for p in predictions])

    prompt = f"""
    The image likely contains: {labels}.
    Dominant color: {color}.
    
    User question: {question}
    
    Provide a clear and helpful answer.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ROUTES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]

    image_bytes = file.read()
    file.seek(0)

    result = query_model(image_bytes)

    if not isinstance(result, list):
        return jsonify(result)

    predictions = sorted(result, key=lambda x: x["score"], reverse=True)[:3]
    color = get_dominant_color(file)

    return jsonify({
        "predictions": predictions,
        "color": color
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json

    predictions = data["predictions"]
    color = data["color"]
    question = data["question"]

    answer = ask_llm(predictions, color, question)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)