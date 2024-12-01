import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import gensim.downloader as api


word_model = api.load('word2vec-google-news-300')
data = pd.read_csv("word_pairs.csv");

similarity_mapping = {"direct": 1.0, "indirect": 0.5, "not-related": 0.0}
data["similarity_value"] = data["similarity_type"].map(similarity_mapping)

def get_word_vector(word):
    return word_model[word] if word in word_model else np.zeros(word_model.vector_size)

def prepare_data(df):
    X1, X2, y = [], [], []
    for _, row in df.iterrows():
        word1_vector = get_word_vector(row["word1"])
        word2_vector = get_word_vector(row["word2"])
        if not np.any(word1_vector) or not np.any(word2_vector):
            continue  # Skip pairs where vectors are missing

        X1.append(word1_vector)
        X2.append(word2_vector)
        y.append(row["similarity_value"])

    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = prepare_data(data)

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42, stratify=y)

X1_train = torch.tensor(X1_train, dtype=torch.float32)
X2_train = torch.tensor(X2_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X1_test = torch.tensor(X1_test, dtype=torch.float32)
X2_test = torch.tensor(X2_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class WordSimilarityModel(nn.Module):
    def __init__(self, vector_size):
        super(WordSimilarityModel, self).__init__()
        self.weight = nn.Parameter(torch.ones(vector_size))  # Weight vector for word1

    def forward(self, word1_vector, word2_vector):
        weighted_word1 = word1_vector * self.weight
        weighted_word2 = word2_vector * self.weight
        
        # Compute the normalized dot product for similarity
        dot_product = torch.sum(weighted_word1 * weighted_word2)
        norm_product = torch.norm(weighted_word1) * torch.norm(weighted_word2)
        
        # Return the normalized dot product similarity
        return dot_product / norm_product
    
vector_size = X1_train.shape[1]
model = WordSimilarityModel(vector_size)

loss_function = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train_model(model, X1_train, X2_train, y_train, epochs=1000):
    model.train()
    pbar = tqdm(range(epochs), desc="Training Epochs", unit="epoch")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # Forward pass
        similarity_pred = model(X1_train, X2_train)
        
        # Compute loss
        loss = loss_function(similarity_pred, y_train)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())

    return model

trained_model = train_model(model, X1_train, X2_train, y_train)

def evaluate_model(model, X1_test, X2_test, y_test):
    model.eval()
    with torch.no_grad():
        similarity_pred = model(X1_test, X2_test)
        similarity_pred = similarity_pred.squeeze()
        squared_errors = (similarity_pred - y_test) ** 2
        
        mse = squared_errors.mean().item()
    return mse

mse = evaluate_model(trained_model, X1_test, X2_test, y_test)
print(f"Mean Squared Error on Test Set: {mse:.4f}")

weight = trained_model.weight.detach().numpy() 
np.save('weight_vector.npy', weight)
