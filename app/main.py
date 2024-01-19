from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

class InputData(BaseModel):
    timestamp: int
    x_axis: float
    y_axis: float
    z_axis: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Human Activity Recognition API!"}

@app.post("/predict")
def predict_activity(data: InputData):
    input_data = [[data.timestamp, data.x_axis, data.y_axis, data.z_axis]]
    
    try:
        model = joblib.load('har_model.joblib')
    except (FileNotFoundError, joblib.JoblibFileError):
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")

    try:
        prediction = model.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting activity: {str(e)}")

    return {"activity": prediction[0]}

