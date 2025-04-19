from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn
import joblib
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

origins = [
    "http://localhost:3000",     # if you're running frontend on React
    "http://10.145.153.156:3000",  # or whatever your frontend IP/port is
    "*"  # You can use "" for testing, but restrict in production
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows frontend origins
    allow_credentials=True,
    allow_methods=["*"],    # Allows all methods, including OPTIONS
    allow_headers=["*"],    # Allows all headers
)



MODEL_PATH = "model_high_recall.h5"  
model = load_model(MODEL_PATH)

SEQ_FEATURES = ['shortage_qty_7d_avg', 'shortage_qty_30d_avg', 'shortage_freq_7d', 'shortage_freq_30d']
STATIC_FEATURES = ['item_encoded', 'day_of_week', 'month', 'quarter', 'year', 'is_weekend',
                   'total_observations', 'historical_shortage_prob', 'avg_shortage_qty', 
                   'max_shortage_qty', 'total_shortage_qty']

class SingleDayInput(BaseModel):
    data: dict  

def preprocess_single_day(input_data: dict):
    df = pd.DataFrame([input_data])
    seq_data = np.tile(df[SEQ_FEATURES].values, (30, 1)).reshape(1, 30, -1)  
    static_data = df[STATIC_FEATURES].values.reshape(1, -1) 
    return seq_data, static_data


xgb_model = joblib.load("xgb_model_precision.pkl")

xgb_features = [
    "item_encoded", "day_of_week", "month", "quarter", "year", "is_weekend",
    "total_observations", "historical_shortage_prob", "avg_shortage_qty",
    "max_shortage_qty", "total_shortage_qty", "shortage_qty_7d_avg",
    "shortage_qty_30d_avg", "shortage_freq_7d", "shortage_freq_30d"
]

class ShortageInput(BaseModel):
    data: Dict[str, float]

THRESHOLD_P = 0.8 

@app.post("/predict_precision")
def predict_xgb(input_json: ShortageInput):
    try:
        input_records = input_json.data

        input_df = pd.DataFrame([input_records])

        input_df = input_df[xgb_features]

        input_array = input_df.to_numpy()

        probabilities = xgb_model.predict_proba(input_array)[:, 1]

        return {
            "probability": float(probabilities[0]),
            "Threshold to Make Decision": THRESHOLD_P
        }

    except Exception as e:
        return {"error": str(e)}

THRESHOLD_R=0.45

@app.post("/predict_recall")
def predict(data: SingleDayInput):
    seq_data, static_data = preprocess_single_day(data.data)
    prediction = model.predict([seq_data, static_data])
    return {
        "probability": float(prediction[0][0]),
        "Threshold to Make Decision": THRESHOLD_R
        }


THRESHOLD_H = 0.6
SEQUENCE_LENGTH = 30

class HybridInput(BaseModel):
    data: Dict[str, float]  # Single record with all features

def preprocess_input(input_data: dict):
    df = pd.DataFrame([input_data])

    seq_array = np.tile(df[SEQ_FEATURES].values, (30, 1)).reshape(1, 30, len(SEQ_FEATURES))

    static_array = df[STATIC_FEATURES].values.reshape(1, -1)

    xgb_array = df[xgb_features].values

    return seq_array, static_array, xgb_array

@app.post("/predict_hybrid")
def predict_hybrid(input_json: HybridInput):
    try:
        input_dict = input_json.data
        seq_array, static_array, xgb_array = preprocess_input(input_dict)

        # Predictions
        lstm_prob = model.predict([seq_array, static_array])[0][0]
        xgb_prob = xgb_model.predict_proba(xgb_array)[:, 1][0]

        # Average probability for hybrid
        final_prob = (lstm_prob + xgb_prob) / 2

        prediction = int(final_prob > THRESHOLD_H)

        return {
            "probability": float(final_prob),
            "Threshold to Make Decision": THRESHOLD_H
        }

    except Exception as e:
        return {"error": str(e)}
    
# Chatbot API development

import chromadb
from langchain.chains import RetrievalQA  
from langchain.vectorstores import Chroma 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="sc_collection")

retriever = Chroma(
    collection_name="sc_collection",
    embedding_function=embeddings
).as_retriever()

llm=CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",config={'max_new_tokens':512,'temperature':0.8})

qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=False,)

@app.post("/rag", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        answer = qa_chain.run(request.query)  
        return QueryResponse(answer=answer, sources=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Shortage Prediction API is running!"}

