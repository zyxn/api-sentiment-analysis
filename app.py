from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from typing import List
from deep_translator import MicrosoftTranslator

# Define the request body structure
class Texts(BaseModel):
    texts: List[str]
    model_name: str

# Global variables to store models and translator
ml_models = {}

# Load the models
def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    global ml_models

    # Load the vectoriser.
    with open('vectoriser-ngram-(1,2).pickle', 'rb') as file:
        ml_models["vectoriser"] = pickle.load(file)
    
    # Load the LR Model v1.
    with open('Sentiment-BNB.pickle', 'rb') as file:
        ml_models["v1"] = pickle.load(file)
        
    # Load the LR Model v2 (assuming there's a second model).
    with open('Sentiment-LR.pickle', 'rb') as file:
        ml_models["v2"] = pickle.load(file)

# Preprocess function placeholder (assuming you have a preprocess function)
def preprocess(text):
    # Add your preprocessing steps here
    return text

# Translate text from Indonesian to English
def translate_texts(texts):
    translator = MicrosoftTranslator(source='id', target='en')
    translated_texts = [translator.translate(text) for text in texts]
    return translated_texts

# Predict function
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for txt, pred in zip(text, sentiment):
        sentiment_label = "Positive" if pred == 1 else "Negative"
        data.append({"text": txt, "sentiment": sentiment_label})
    
    return data

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    load_models()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict/")
async def get_sentiment(texts: Texts):
    try:
        # Translate texts from Indonesian to English
        translated_texts = translate_texts(texts.texts)

        # Select the model
        model = ml_models.get(texts.model_name)
        vectoriser = ml_models.get("vectoriser")
        if model is None or vectoriser is None:
            raise HTTPException(status_code=400, detail="Invalid model name. Choose 'v1' or 'v2'.")
        
        # Predict the sentiment
        result = predict(vectoriser, model, translated_texts)
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
