
# Scam Call Detection API

## Project Description
This project provides a backend API for detecting scam calls. It utilizes a hybrid model approach, combining a fine-tuned `distilbert-base-multilingual-cased` model for extracting contextual embeddings and an XGBoost classifier for final classification. This deployment package specifically uses the `float16` precision version of the DistilBERT model, optimized for a balance of size and performance.

## Setup and Installation

1.  **Clone the repository (if applicable) or navigate to the project directory:**
    ```bash
    cd .
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    All necessary Python libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the FastAPI Application

To run the application locally, you can use `uvicorn`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Once running, the API documentation will be available at `http://0.0.0.0:8000/docs`.

## API Endpoints

### 1. Health Check
Checks if the API and models are loaded and ready.

-   **Endpoint:** `/health`
-   **Method:** `GET`
-   **Response:**
    ```json
    {
      "status": "ok",
      "model_loaded": true
    }
    ```

### 2. Scam Prediction
Predicts whether a given text input is a scam or not.

-   **Endpoint:** `/api/v1/predict`
-   **Method:** `POST`
-   **Request Body:**
    ```json
    {
      "text": "Hello, this is your bank. Your account has been compromised."
    }
    ```
-   **Response:**
    ```json
    {
      "label": 1,      // 1 for scam, 0 for not scam
      "probability": 0.98 // Probability of being a scam
    }
    ```

## Model Architecture and Deployment Notes

### DistilBERT Model (Float16 Version)
-   **Base Model:** `distilbert-base-multilingual-cased`
-   **Fine-tuning:** The original model was fine-tuned for sequence classification on the scam detection dataset.
-   **Precision Conversion:** The `float32` fine-tuned model was converted to `float16` precision, reducing its size to approximately 258.15 MB.
-   **Splitting Strategy:** The `float16` model's state dictionary was serialized into a byte stream and split into 11 parts, each less than 25MB. These parts are located in the same directory as `model.py`.
-   **Reconstruction:** The `model.py` contains a `reconstruct_float16_model` function that reads these split parts, re-assembles them, and loads the state dictionary into a `float16` `AutoModelForSequenceClassification` instance.

### XGBoost Classifier
-   **Role:** Acts as a meta-classifier, taking the CLS embeddings extracted from the DistilBERT model as features.
-   **Training:** Trained on the CLS embeddings of the training data.

### Model Reconstruction for Deployment
In `model.py`, the `ScamDetectorModel` class is configured to directly load the `float16` DistilBERT model and the XGBoost classifier from the current directory, streamlining the deployment process.

### Stored Models and Files (within this directory: `./`)
-   `float16_distilbert_part_*.bin_part`: Split parts of the *float16* DistilBERT model.
-   `xgboost_model.json`: The trained XGBoost classifier.
-   `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`, `config.json`: Files related to the DistilBERT tokenizer and model configuration.
-   `model.py`: Contains the `ScamDetectorModel` class for loading and inference.
-   `routes.py`: Defines the FastAPI endpoints.
-   `app.py`: The main FastAPI application.
-   `requirements.txt`: Project dependencies.
-   `.gitignore`: Git ignore file.
