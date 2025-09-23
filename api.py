"""
Email Reply Classification API

FastAPI service for classifying email replies as positive, negative, or neutral.
Uses the trained Logistic Regression model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re
import os
from typing import Dict, Any
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Email Reply Classification API",
    description="Classify email replies as positive, negative, or neutral",
    version="1.0.0"
)

# Global variables for model components
model = None
vectorizer = None
label_encoder = None

# Request/Response models
class EmailRequest(BaseModel):
    text: str

class EmailResponse(BaseModel):
    label: str
    confidence: float

def clean_text(text):
    """Clean and preprocess text data (same as training pipeline)"""
    if pd.isna(text) or not text:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Handle common abbreviations and contractions
    text = re.sub(r'\bu\b', 'you', text)
    text = re.sub(r'\bw/\b', 'with', text)
    text = re.sub(r'\bplz\b', 'please', text)
    text = re.sub(r'\blets\b', 'let us', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'d", ' would', text)
    text = re.sub(r"'m", ' am', text)
    
    # Remove excessive punctuation but keep some for context
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[,]{2,}', ',', text)
    
    return text

def train_and_save_model():
    """Train the model and save it for API use"""
    print("Training model for API deployment...")
    
    # Import training functions
    from email_reply_classification import (
        load_and_explore_data, preprocess_data, train_baseline_models
    )
    from sklearn.model_selection import train_test_split
    
    # Load and preprocess data
    df = pd.read_csv('reply_classification_dataset.csv')
    
    # Clean text
    df['cleaned_reply'] = df['reply'].apply(clean_text)
    
    # Standardize labels
    df['label'] = df['label'].str.lower()
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    # Remove empty text
    df = df[df['cleaned_reply'].str.len() > 0].copy()
    
    # Split data
    X = df['cleaned_reply']
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train vectorizer and model
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    lr_model.fit(X_train_tfidf, y_train)
    
    # Save model components
    joblib.dump(lr_model, 'email_classifier_model.pkl')
    joblib.dump(vectorizer, 'email_classifier_vectorizer.pkl')
    joblib.dump(label_encoder, 'email_classifier_label_encoder.pkl')
    
    print("Model trained and saved successfully!")
    return lr_model, vectorizer, label_encoder

def load_model():
    """Load the trained model components"""
    global model, vectorizer, label_encoder
    
    try:
        model = joblib.load('email_classifier_model.pkl')
        vectorizer = joblib.load('email_classifier_vectorizer.pkl')
        label_encoder = joblib.load('email_classifier_label_encoder.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        model, vectorizer, label_encoder = train_and_save_model()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Email Reply Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Classify email reply text",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": label_encoder.classes_.tolist()
    }

@app.post("/predict", response_model=EmailResponse)
async def predict_email(request: EmailRequest) -> EmailResponse:
    """
    Classify email reply text
    
    Args:
        request: EmailRequest with text field
        
    Returns:
        EmailResponse with label and confidence
    """
    if model is None or vectorizer is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")
    
    try:
        # Clean and preprocess text
        cleaned_text = clean_text(request.text)
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text contains no valid content after cleaning")
        
        # Vectorize text
        text_vector = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        # Get label and confidence
        label = label_encoder.inverse_transform([prediction])[0]
        confidence = float(prediction_proba[prediction])
        
        return EmailResponse(
            label=label,
            confidence=round(confidence, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
