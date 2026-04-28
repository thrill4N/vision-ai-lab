from flask import Flask, render_template, request, jsonify
import requests, os, json
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import faiss
from openai import OpenAI

load_dotenv()  # Load environment variables from .env file

# Set up HuggingFace API headers with authentication token
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}"}
# Load model URLs from environment
HF_CLASSIFIER = os.getenv("HF_CLASSIFIER_URL")  # Vision model for object detection
HF_EMBEDDING = os.getenv("HF_EMBEDDING_URL")  # CLIP model for image embeddings
HF_CAPTION = os.getenv("HF_CAPTION_URL")  # BLIP model for image captions

# Initialize OpenAI client for LLM interactions
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# Initialize FAISS vector database for similarity search
DIM = 512  # Dimension of CLIP embeddings
index = faiss.IndexFlatL2(DIM)  # L2 distance metric for similarity

chat_history = []  # Store conversation history for multi-turn interactions

# Make API calls to HuggingFace models with error handling
def call_api(url, data):
    res = requests.post(url, headers=HF_HEADERS, data=data)

    print("STATUS:", res.status_code)  # Debug: print response status
    print("RESPONSE:", res.text[:200])  # Debug: print first 200 chars of response

    # Check if response is empty
    if not res.content:
        return {"error": "Empty response"}

    # Try to parse JSON, return error if fails
    try:
        return res.json()
    except:
        return {"error": res.text[:100]}

def get_color(file):
    # Open image and convert to RGB, resize for faster processing
    img = Image.open(file).convert("RGB").resize((100, 100))
    # Flatten pixel data into array of RGB values
    pixels = np.array(img).reshape(-1, 3)
    # Calculate average RGB values to find dominant color
    avg = pixels.mean(axis=0).astype(int)
    return f"RGB({avg[0]}, {avg[1]}, {avg[2]})"

def get_caption(image_bytes):
    # Call BLIP model to generate image caption
    result = call_api(HF_CAPTION, image_bytes)
    # Extract caption text from response (API returns list of results)
    if isinstance(result, list):
        return result[0]["generated_text"]
    return "No caption available"

def add_to_index(embedding):
    # Extract embedding if API returned nested list format [[...]]  
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]
    # Convert to float32 numpy array and add to FAISS index
    vec = np.array([embedding]).astype("float32")
    index.add(vec)  # Store for future similarity comparisons

def search_index(embedding):
    # Return early if no previous images in index
    if index.ntotal == 0:
        return "No previous images", 0

    # Extract embedding if API returned nested list format [[...]]
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]
    # Convert to float32 and search for most similar image (k=1)
    vec = np.array([embedding]).astype("float32")
    D, _ = index.search(vec, 1)  # D contains distance to nearest neighbor
    d = float(D[0][0])  # Extract distance value

    # Interpret similarity based on distance threshold
    if d < 0.2:
        return "Very similar image found", d
    elif d < 0.5:
        return "Some similarity detected", d
    return "Different from previous images", d

def ask_llm(context, question):
    # Add user question to conversation history
    chat_history.append({"role": "user", "content": question})

    # Call OpenAI API with image context and conversation history
    res = client.chat.completions.create(
        model="gpt-4o-mini",  # Use GPT-4 mini model for cost efficiency
        messages=[{"role": "system", "content": context}] + chat_history  # Include image data as system context
    )

    # Extract answer from response and add to history
    answer = res.choices[0].message.content
    chat_history.append({"role": "assistant", "content": answer})
    return answer

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")  # Serve the main UI

# API endpoint to analyze uploaded image
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]  # Get uploaded image

    # Read image bytes twice (file pointer reset for reuse)
    image_bytes = file.read()
    file.seek(0)  # Reset pointer for PIL processing

    # Call HuggingFace models: classifier for objects, CLIP for embeddings
    predictions = call_api(HF_CLASSIFIER, image_bytes)
    embedding = call_api(HF_EMBEDDING, image_bytes)

    # Check for API errors
    if "error" in predictions:
        return jsonify({"error": predictions["error"]})

    if "error" in embedding:
        return jsonify({"error": embedding["error"]})

    # Sort predictions by confidence score, keep top 3
    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)[:3]

    # Extract image features
    color = get_color(file)  # Get dominant color
    caption = get_caption(image_bytes)  # Generate caption

    # Search for similar images and add current image to index
    comparison, similarity = search_index(embedding)
    add_to_index(embedding)

    # Return all analysis results as JSON
    return jsonify({
        "predictions": predictions,  # Detected objects
        "caption": caption,  # Image description
        "color": color,  # Dominant color
        "comparison": comparison,  # Similarity to previous images
        "similarity": similarity  # Similarity score
    })

# API endpoint for asking questions about the analyzed image
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json  # Get request payload

    # Construct context string with image information for LLM
    context = f"""
    Caption: {data['caption']}
    Objects: {[p['label'] for p in data['predictions']]}
    Color: {data['color']}
    """

    # Call LLM with image context and user question
    answer = ask_llm(context, data["question"])
    return jsonify({"answer": answer})

# Run Flask app in debug mode for development
if __name__ == "__main__":
    app.run(debug=True)
