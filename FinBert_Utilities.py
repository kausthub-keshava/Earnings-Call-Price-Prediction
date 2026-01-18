import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "yiyanghkust/finbert-pretrain"  # finBERT model for embeddings
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
encoder.eval()  # freeze FinBERT weights

# function to break transcripts into chunks of max 512 tokens
def chunks(text, max_tokens,overlap,tokenizer=tokenizer):
    assert 0<=overlap < max_tokens, "overlap must be smaller than max_tokens"
    tokens = tokenizer(text,add_special_tokens=False,return_attention_mask=False,return_tensors=None,truncation=False)['input_ids']
    token_length = len(tokens)
    chunks=[]
    start=0
    end=max_tokens
    while start < token_length:
        chunk = tokens[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = min(start + max_tokens,token_length)
        if end==token_length:
            break
    return chunks    

# function to get FinBERT embeddings for a given chunk of text
def get_finbert_embedding(chunk,encoder=encoder,tokenizer=tokenizer,device=device):
    inputs=tokenizer.prepare_for_model(chunk,return_tensors='pt',padding='max_length',max_length=512,truncation=True,return_attention_mask=True,add_special_tokens=True)
    inputs= {k:v.to(device) for k,v in inputs.items()}

    if inputs["input_ids"].dim() == 1:
        inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
    if inputs["attention_mask"].dim() == 1:
        inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)
    if "token_type_ids" in inputs and inputs["token_type_ids"].dim() == 1:
        inputs["token_type_ids"] = inputs["token_type_ids"].unsqueeze(0)
    with torch.no_grad():
        chunk_embeddings=encoder(**inputs)   
    return chunk_embeddings.last_hidden_state, inputs['attention_mask']    

# function to get one vector per chunk by attention pooling
def attention_pooling(chunk_embeddings, attention_mask,attention_vector):
    score=chunk_embeddings@attention_vector
    score = score.masked_fill(attention_mask==0,-1e9)
    weights = torch.softmax(score, dim=1)
    chunk_vector=(weights.unsqueeze(-1)*chunk_embeddings).sum(dim=1)
    return chunk_vector

# function to get one vector per chunk by mean pooling
def mean_pooling(embeddings, attention_mask):
    attention_mask = attention_mask.unsqueeze(-1)
    masked_embeddings = embeddings * attention_mask
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    mean_pooled_embedding = sum_embeddings / sum_mask
    return mean_pooled_embedding

# function to input transcript and get one Finbert embedding by chunking, attention pooling and taking mean over chunks
def get_transcript_embeddings(transcript,attention_pool_v,overlap,encoder=encoder,tokenizer=tokenizer,device=device,max_tokens=512):
    transcript_chunks = chunks(transcript, max_tokens=max_tokens, overlap=overlap,tokenizer=tokenizer)
    transcript_attn_vector = []
    transcript_mean_vector = []
    for chunk in transcript_chunks:
        embedding, attention_mask = get_finbert_embedding(chunk, encoder=encoder, tokenizer=tokenizer, device=device)
        chunk_attn_vector= attention_pooling(embedding, attention_mask, attention_pool_v)
        transcript_attn_vector.append(chunk_attn_vector)
        chunk_mean_vector= mean_pooling(embedding, attention_mask)
        transcript_mean_vector.append(chunk_mean_vector)

    return torch.stack(transcript_attn_vector,dim=0).mean(dim=0), torch.stack(transcript_mean_vector,dim=0).mean(dim=0)

import torch.nn as nn
import torch.nn.functional as F

class FinBERT_Classifier(nn.Module):
    def __init__(self, encoder, attention_pool_dim=768, hidden_dim=256, output_dim=1, dropout=0.2):
        super(FinBERT_Classifier, self).__init__()
        self.encoder = encoder
        self.attention_vector = nn.Parameter(torch.randn(attention_pool_dim))
        self.fc1 = nn.Linear(attention_pool_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attn_vector):
        x = F.relu(self.fc1(attn_vector))
        x = self.dropout(x)
        logit = self.fc2(x).squeeze(-1)
        return logit

# Forward pass 
loss_fn = nn.BCEWithLogitsLoss()

def forward_one(model, transcript, y, overlap=50, max_tokens=512):
    # 1) transcript -> pooled embedding using model.attention_vector
    attn_vec, _ = get_transcript_embeddings(
        transcript,
        attention_pool_v=model.attention_vector,
        overlap=overlap,
        encoder=model.encoder,
        tokenizer=tokenizer,
        device=device,
        max_tokens=max_tokens
    )  # shape (1,768)

    # 2) embedding -> logit
    logit = model(attn_vec)  # shape (1,) basically

    # 3) loss
    y_t = torch.tensor([float(y)], device=device)  # shape (1,)
    loss = loss_fn(logit.view(-1), y_t)
    return loss, logit

@torch.no_grad()
def evaluate(model, data, overlap=50, max_tokens=512):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for transcript, y in data:
        loss, logit = forward_one(model, transcript, y, overlap=overlap, max_tokens=max_tokens)
        total_loss += float(loss.item())

        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).long().item()
        correct += int(pred == int(y))
        n += 1

    return total_loss / max(1, n), correct / max(1, n)
