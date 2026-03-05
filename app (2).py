
from fastapi import FastAPI
from routes import router

app = FastAPI(
    title="Scam Call Detection API",
    description="API for detecting scam calls using a fine-tuned DistilBERT and XGBoost model.",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Scam Call Detection API! Visit /docs for API documentation."}
