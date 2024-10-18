from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import torch

app = FastAPI()

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('./distilbert-qa-model')  # Adjust path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

class QARequest(BaseModel):
    question: str
    context: str

@app.post("/predict")
def predict(qa_request: QARequest):
    inputs = tokenizer(qa_request.question, qa_request.context, max_length=256, truncation=True, padding="max_length", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)

    input_ids = inputs['input_ids'].squeeze().tolist()
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens)

    return {"answer": answer}
