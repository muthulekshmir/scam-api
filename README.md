
# Scam Call Detection API

## Project Description
This project provides a backend API for detecting scam calls. It utilizes a hybrid model approach, combining a fine-tuned `distilbert-base-multilingual-cased` model for extracting contextual embeddings and an XGBoost classifier for final classification. The DistilBERT model has been quantized and split into multiple parts (each <25MB) to optimize for deployment in environments with size constraints, such as serverless functions or edge devices.

## Setup and Installation

1.  **Clone the repository (if applicable) or navigate to the project directory:**
    ```bash
    cd /content/drive/MyDrive/scamapi/
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scriptsctivate`
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

### DistilBERT Model
-   **Base Model:** `distilbert-base-multilingual-cased`
-   **Fine-tuning:** The model was fine-tuned for sequence classification on the scam detection dataset.
-   **Quantization:** The fine-tuned DistilBERT model was dynamically quantized to `qint8` to reduce its memory footprint and speed up inference on CPU.
-   **Splitting Strategy:** To meet potential deployment constraints (e.g., individual file size limits), the quantized model's state dictionary was serialized into a byte stream and then split into 16 parts, each less than 25MB. This allows for easier transfer and storage.

### XGBoost Classifier
-   **Role:** Acts as a meta-classifier, taking the CLS embeddings extracted from the DistilBERT model as features.
-   **Training:** Trained on the CLS embeddings of the training data.

### Model Reconstruction for Deployment
In `model.py`, a `reconstruct_quantized_model` function is implemented. This function reads all the split parts of the quantized DistilBERT model from the file system, re-assembles them into a complete state dictionary, and then loads this state dictionary into a dynamically quantized `AutoModelForSequenceClassification` instance. This mechanism ensures that the model can be loaded and used correctly in the FastAPI application.

### Stored Models and Files
All models and backend files are saved in the `/content/drive/MyDrive/scamapi/` directory:
-   `quantized_distilbert_part_*.pt`: Split parts of the quantized DistilBERT model.
-   `xgboost_model.json`: The trained XGBoost classifier.
-   `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`, `config.json`: Files related to the DistilBERT tokenizer and model configuration.
-   `model.py`: Contains the `ScamDetectorModel` class for loading and inference.
-   `routes.py`: Defines the FastAPI endpoints.
-   `app.py`: The main FastAPI application.
-   `requirements.txt`: Project dependencies.
-   `.gitignore`: Git ignore file.
