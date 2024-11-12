from sklearn.svm import OneClassSVM
import numpy as np
from transformers import BertModel, BertTokenizer
import torch

# Define the device - use GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


bert_model = "google-bert/bert-base-multilingual-cased"
# bert_model = 'google-bert/bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model).to(device)

def convert_terms_to_embeddings(to_tokenize_list, use_cls_token=False):
    if to_tokenize_list is None or len(to_tokenize_list) == 0:
        return []
    inputs = tokenizer(to_tokenize_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokenized_inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    if use_cls_token:
        # Use the CLS token output as the embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token is at position 0
        numpy_tensors = cls_embedding.cpu().numpy()
    else:
        # Use the mean of the last hidden state as the embedding
        mean_embedding = outputs.last_hidden_state.mean(dim=1)
        numpy_tensors = mean_embedding.cpu().numpy()

    return numpy_tensors
