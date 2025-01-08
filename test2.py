import os
print(os.getcwd()) 
os.chdir("/Users/Andrew/Desktop/HTML/Bergen Economic Data EDA") 
print(os.getcwd()) 

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load CSV
df = pd.read_csv("bergen_county_1180_questions_answers.csv")  # columns: Question, Answer

# Load an embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = model.encode(df['Questions'].tolist())

# Normalize or keep embeddings as is for similarity search
question_embeddings = question_embeddings / np.linalg.norm(question_embeddings, axis=1, keepdims=True)

# Prepare Flask app
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "AI Chatbot Backend is running."

@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.json.get("question", "")
    user_embedding = model.encode([user_question])[0]
    user_embedding = user_embedding / np.linalg.norm(user_embedding)

    # Compute cosine similarity
    similarities = np.dot(question_embeddings, user_embedding)
    best_idx = np.argmax(similarities)
    best_answer = df['Answers'].iloc[best_idx]
    
    return jsonify({"answer": best_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))




