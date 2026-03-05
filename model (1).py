
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
import os
import io

MODEL_NAME = 'distilbert-base-multilingual-cased'
NUM_LABELS = 2
PROJECT_DIR = "/content/drive/MyDrive/scamapi/"
# Placeholder for NUM_PARTS_QUANTIZED, will be replaced with actual value
NUM_PARTS_QUANTIZED_PLACEHOLDER = 16 

def reconstruct_quantized_model(output_dir, num_parts, model_name, num_labels):
    # Re-assemble the byte stream from individual parts
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

class ScamDetectorModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(PROJECT_DIR)
        self.distilbert_model = self.load_distilbert_model()
        self.xgboost_model = self.load_xgboost_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distilbert_model.to(self.device)
        self.distilbert_model.eval()

    def load_distilbert_model(self):
        # Use the actual NUM_PARTS_QUANTIZED from the notebook's execution context
        num_parts = NUM_PARTS_QUANTIZED_PLACEHOLDER 
        if num_parts == -1: # Fallback if placeholder isn't replaced
            print("Warning: NUM_PARTS_QUANTIZED not set. Defaulting to 16.")
            num_parts = 16 # Adjust if your actual split differs
        
        model = reconstruct_quantized_model(
            PROJECT_DIR, num_parts, MODEL_NAME, NUM_LABELS
        )
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
