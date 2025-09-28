from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend_model_improved import improved_risk_system



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rockfall Prediction Endpoint
@app.post("/api/predict")
async def predict_risk(values: dict = Body(...)):
    try:
        # 1️⃣ Get risk score from ML model
        risk_score = improved_risk_system.calculate_risk(values)

        # 2️⃣ Determine category and recommendation
        risk_category, recommendation = improved_risk_system.get_category(risk_score)

        # 3️⃣ Generate confidence score (for demo, use inverse of distance from mid)
        confidence_score = round(1 - abs(0.5 - risk_score), 2)

        # 4️⃣ Generate recommendations (you can make these smarter later)
        recommendations = [
            recommendation,
            "Monitor slope conditions closely" if risk_score > 0.5 else "Routine inspection sufficient",
            "Reduce blasting frequency" if values["days_since_blast"] < 3 else "Continue standard operations"
        ]

        # 5️⃣ Return structured response
        return {
            "prediction": {
                "risk_score": risk_score,
                "risk_category": risk_category.capitalize(),
                "confidence_score": confidence_score,
                "recommendations": recommendations
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
