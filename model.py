
import os
import torch
import numpy as np
import xgboost as xgb

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizerFast


MODEL_FILE = "distilbert_model.bin"


# -----------------------------
# Merge DistilBERT split files
# -----------------------------
def merge_model_parts():
    if os.path.exists(MODEL_FILE):
        return

    print("Merging DistilBERT model parts...")

    with open(MODEL_FILE, "wb") as outfile:
        i = 1
        while True:
            part_name = f"float16_distilbert_part_{i}.bin_part"

            if not os.path.exists(part_name):
                break

            with open(part_name, "rb") as infile:
                outfile.write(infile.read())

            i += 1

    print("Model merge completed")


# -----------------------------
# Load models only once
# -----------------------------
class ScamModel:

    def __init__(self):

        merge_model_parts()

        print("Loading tokenizer...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(".", local_files_only=True)

        print("Loading DistilBERT config...")
        config = DistilBertConfig.from_pretrained("config.json")

        print("Loading DistilBERT model...")
        self.bert = DistilBertModel(config)

        state_dict = torch.load(MODEL_FILE, map_location="cpu")
        self.bert.load_state_dict(state_dict)

        self.bert.eval()

        print("Loading XGBoost model...")
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model("xgboost_model.json")

        print("SmartCall model loaded successfully")


    # -----------------------------
    # Get embedding from DistilBERT
    # -----------------------------
    def get_embedding(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.bert(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].numpy()

        return embedding


    # -----------------------------
    # Prediction pipeline
    # -----------------------------
    def predict(self, text):

        embedding = self.get_embedding(text)

        prob = self.xgb.predict_proba(embedding)[0][1]

        if prob > 0.5:
            label = "scam"
        else:
            label = "not scam"

        return label, float(prob)


# -----------------------------
# Global model instance
# -----------------------------
scam_model = ScamModel()
