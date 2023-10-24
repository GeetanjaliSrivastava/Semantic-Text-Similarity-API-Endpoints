from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI()

class RequestModel(BaseModel):
    text1: str
    text2: str

# Load the SentenceTransformer model outside of the API endpoint function
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')

@app.post("/semantic_similarity")
def semantic_similarity(request_data: RequestModel):
    text1 = request_data.text1
    text2 = request_data.text2

    # Encode the sentences using the model
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0].view(1, -1), embeddings[1].view(1, -1))

    # Extract the similarity score as a float
    similarity_score = similarity_score.item()

    return {'similarity_score': similarity_score}
