
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
import os
import io

MODEL_NAME = 'distilbert-base-multilingual-cased'
NUM_LABELS = 2
PROJECT_DIR = "/content/drive/MyDrive/scamapi/"
# Placeholders for NUM_PARTS_QUANTIZED and NUM_PARTS_UNQUANTIZED, will be replaced with actual values
NUM_PARTS_QUANTIZED_PLACEHOLDER = 16
NUM_PARTS_UNQUANTIZED_PLACEHOLDER = 21

def reconstruct_quantized_model(output_dir, num_parts, model_name, num_labels):
    reconstructed_bytes = io.BytesIO()
    for i in range(num_parts):
        part_file_path = os.path.join(output_dir, f'quantized_distilbert_part_{i+1}.pt')
        with open(part_file_path, 'rb') as f:
            reconstructed_bytes.write(f.read())
    reconstructed_bytes.seek(0)
    reconstructed_state_dict = torch.load(reconstructed_bytes, map_location=torch.device('cpu'))
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    reconstructed_model = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    reconstructed_model.load_state_dict(reconstructed_state_dict)
    return reconstructed_model

def reconstruct_unquantized_model(output_dir, num_parts, model_name, num_labels):
    reconstructed_bytes = io.BytesIO()
    for i in range(num_parts):
        part_file_path = os.path.join(output_dir, f'distilbert_part_{i+1}.bin_part')
        with open(part_file_path, 'rb') as f:
            reconstructed_bytes.write(f.read())
    reconstructed_bytes.seek(0)
    reconstructed_state_dict = torch.load(reconstructed_bytes, map_location=torch.device('cpu'))
    reconstructed_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    reconstructed_model.load_state_dict(reconstructed_state_dict)
    return reconstructed_model

class ScamDetectorModel:
    def __init__(self, use_quantized_model: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(PROJECT_DIR)
        self.use_quantized_model = use_quantized_model
        self.distilbert_model = self.load_distilbert_model()
        self.xgboost_model = self.load_xgboost_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distilbert_model.to(self.device)
        self.distilbert_model.eval()

    def load_distilbert_model(self):
        if self.use_quantized_model:
            print("Loading (and reconstructing) quantized DistilBERT model...")
            num_parts = NUM_PARTS_QUANTIZED_PLACEHOLDER
            if num_parts == -1: print("Warning: NUM_PARTS_QUANTIZED not set. Defaulting to 16."); num_parts = 16 # Fallback
            model = reconstruct_quantized_model(
                PROJECT_DIR, num_parts, MODEL_NAME, NUM_LABELS
            )
            print("Quantized DistilBERT model loaded.")
        else:
            print("Loading (and reconstructing) unquantized DistilBERT model...")
            num_parts = NUM_PARTS_UNQUANTIZED_PLACEHOLDER
            if num_parts == -1: print("Warning: NUM_PARTS_UNQUANTIZED not set. Defaulting to 16."); num_parts = 16 # Fallback
            model = reconstruct_unquantized_model(
                PROJECT_DIR, num_parts, MODEL_NAME, NUM_LABELS
            )
            print("Unquantized DistilBERT model loaded.")
        return model

    def load_xgboost_model(self):
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.load_model(os.path.join(PROJECT_DIR, 'xgboost_model.json'))
        return xgb_model

    def get_cls_embedding(self, text):
        encoding = self.tokenizer(
            str(text),
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.distilbert_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        return cls_embedding

    def predict_scam(self, text: str):
        cls_embedding = self.get_cls_embedding(text)
        prediction = self.xgboost_model.predict(cls_embedding)
        probability = self.xgboost_model.predict_proba(cls_embedding)[:, 1]
        return {"label": int(prediction[0]), "probability": float(probability[0])}
