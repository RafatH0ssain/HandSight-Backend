from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import io
import sys

# Import our custom Vision Engine
from vision_engine import HandwritingAnalyzer

# GLOBAL VARIABLE TO HOLD THE MODEL
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML model when the app starts.
    Clean it up when the app stops.
    """
    try:
        # Load the model you created in Step 1
        model = joblib.load('personality_engine.pkl')
        ml_models["personality"] = model
        print("✅ SUCCESS: Personality Model Loaded!")
    except FileNotFoundError:
        print("❌ ERROR: 'personality_engine.pkl' not found.")
        print("   -> Did you run the training script? Is the file in this folder?")
    except Exception as e:
        print(f"❌ ERROR: Could not load model: {e}")
    
    yield
    
    # Clean up (if necessary)
    ml_models.clear()

# Initialize App with Lifespan
app = FastAPI(lifespan=lifespan)

# CORS: Allow your future frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel to talk to Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Handwriting Analysis API is Running"}

@app.post("/analyze")
async def analyze_handwriting(file: UploadFile = File(...)):
    # 1. Check if model is loaded
    if "personality" not in ml_models:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # 2. Validate File Type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG.")

    try:
        # 3. Read image bytes
        image_bytes = await file.read()
        
        # 4. Pass to Vision Engine
        analyzer = HandwritingAnalyzer(image_bytes)
        features = analyzer.extract_features()
        
        # 5. Format for Model (Must match training columns exactly)
        # Note: We wrap it in a DataFrame because sklearn expects a 2D array/DataFrame
        input_df = pd.DataFrame([features])
        
        # 6. Predict using the ML Model
        # The model returns an array of 5 values (Openness, Conscientiousness, etc.)
        predictions = ml_models["personality"].predict(input_df)[0]
        
        # 7. Construct Response
        return {
            "status": "success",
            "traits": {
                "Openness": round(predictions[0], 2),
                "Conscientiousness": round(predictions[1], 2),
                "Extraversion": round(predictions[2], 2),
                "Agreeableness": round(predictions[3], 2),
                "Neuroticism": round(predictions[4], 2),
            },
            "analysis_reasoning": {
                "slant_score": features["Feature_1"],
                "spacing_score": features["Feature_2"],
                "pressure_score": features["Feature_3"],
                "size_score": features["Feature_8"]
            }
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)