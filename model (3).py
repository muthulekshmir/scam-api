
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
import os
import io

MODEL_NAME = 'distilbert-base-multilingual-cased'
NUM_LABELS = 2
NUM_PARTS_FLOAT16 = 11

def reconstruct_float16_model(output_dir, num_parts, model_name, num_labels):
    reconstructed_bytes_buffer = io.BytesIO()
    for i in range(1, num_parts + 1):
        part_file_path = os.path.join(output_dir, f'float16_distilbert_part_{i}.bin_part')
        with open(part_file_path, 'rb') as f:
            reconstructed_bytes_buffer.write(f.read())
    reconstructed_bytes_buffer.seek(0)
    reconstructed_state_dict = torch.load(reconstructed_bytes_buffer, map_location=torch.device('cpu'))
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    reconstructed_model = base_model.half()
    reconstructed_model.load_state_dict(reconstructed_state_dict)
    return reconstructed_model

class ScamDetectorModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(".") 
        self.distilbert_model = self.load_distilbert_model()
        self.xgboost_model = self.load_xgboost_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distilbert_model.to(self.device)
        self.distilbert_model.eval()

    def load_distilbert_model(self):
        print("Loading (and reconstructing) float16 DistilBERT model...")
        model = reconstruct_float16_model(
            ".", NUM_PARTS_FLOAT16, MODEL_NAME, NUM_LABELS
        )
        print("Float16 DistilBERT model loaded.")
        return model

    def load_xgboost_model(self):
        print("Loading XGBoost model...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.load_model(os.path.join(".", 'xgboost_model.json'))
        print("XGBoost model loaded.")
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
