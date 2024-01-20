from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class InputData(BaseModel):
    timestamp: int
    x_axis: float
    y_axis: float
    z_axis: float

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict_activity(request: Request, timestamp: int = Form(...), x_axis: float = Form(...), y_axis: float = Form(...), z_axis: float = Form(...)):
    input_data = [[timestamp, x_axis, y_axis, z_axis]]
    
    try:
        model = joblib.load('har_model.joblib')
    except (FileNotFoundError, joblib.JoblibFileError):
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")

    try:
        prediction = model.predict(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting activity: {str(e)}")

    return templates.TemplateResponse("result.html", {"request": request, "activity": prediction[0]})
