# Earnings-Call-Price-Prediction
NLP model that consumes language used in earnings calls and predict next 3/5/10 day price moves

Consume Earnings call transcripts data from Kaggle: 

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ramssvimala/earning-call-transcripts")

print("Path to dataset files:", path)
