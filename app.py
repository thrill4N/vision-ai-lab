from flask import Flask, render_template, request, jsonify
import requests, os, json
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import faiss
from openai import OpenAI

load_dotenv()

HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}"}
HF_CLASSIFIER = os.getenv("HF_CLASSIFIER_URL")
HF_EMBEDDING = os.getenv("HF_EMBEDDING_URL")
HF_CAPTION = os.getenv("HF_CAPTION_URL")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# FAISS
DIM = 512
index = faiss.IndexFlatL2(DIM)

chat_history = []

# SAFE API CALL
def call_api(url, data):
    res = requests.post(url, headers=HF_HEADERS, data=data)

    print("STATUS:", res.status_code)
    print("RESPONSE:", res.text[:200])

    if not res.content:
        return {"error": "Empty response"}

    try:
        return res.json()
    except:
        return {"error": res.text[:100]}

def get_color(file):
    img = Image.open(file).convert("RGB").resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    avg = pixels.mean(axis=0).astype(int)
    return f"RGB({avg[0]}, {avg[1]}, {avg[2]})"

def get_caption(image_bytes):
    result = call_api(HF_CAPTION, image_bytes)
    if isinstance(result, list):
        return result[0]["generated_text"]
    return "No caption available"

def add_to_index(embedding):
    # Extract embedding if it's a list of embeddings
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]
    vec = np.array([embedding]).astype("float32")
    index.add(vec)

def search_index(embedding):
    if index.ntotal == 0:
        return "No previous images", 0

    # Extract embedding if it's a list of embeddings
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]
    vec = np.array([embedding]).astype("float32")
    D, _ = index.search(vec, 1)
    d = float(D[0][0])

    if d < 0.2:
        return "Very similar image found", d
    elif d < 0.5:
        return "Some similarity detected", d
    return "Different from previous images", d

def ask_llm(context, question):
    chat_history.append({"role": "user", "content": question})

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": context}] + chat_history
    )

    answer = res.choices[0].message.content
    chat_history.append({"role": "assistant", "content": answer})
    return answer

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]

    image_bytes = file.read()
    file.seek(0)

    predictions = call_api(HF_CLASSIFIER, image_bytes)
    embedding = call_api(HF_EMBEDDING, image_bytes)

    if "error" in predictions:
        return jsonify({"error": predictions["error"]})

    if "error" in embedding:
        return jsonify({"error": embedding["error"]})

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]

    color = get_color(file)
    caption = get_caption(image_bytes)

    comparison, similarity = search_index(embedding)
    add_to_index(embedding)

    return jsonify({
        "predictions": predictions,
        "caption": caption,
        "color": color,
        "comparison": comparison,
        "similarity": similarity
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json

    context = f"""
    Caption: {data['caption']}
    Objects: {[p['label'] for p in data['predictions']]}
    Color: {data['color']}
    """

    answer = ask_llm(context, data["question"])
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)