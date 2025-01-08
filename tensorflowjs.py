import tensorflow_hub as hub
import tensorflowjs as tfjs
import json
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

print(os.getcwd()) 
os.chdir("/Users/Andrew/Desktop/HTML/.venv/Bergen Economic Data EDA") 
print(os.getcwd()) 


# Load Universal Sentence Encoder (USE)
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Save for TensorFlow.js
tfjs.converters.save_keras_model(model, "./tfjs_model")
