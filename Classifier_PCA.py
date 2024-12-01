import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gensim.downloader as api

# Load word2vec model and dataset
word_model = api.load('word2vec-google-news-300')
data = pd.read_csv("word_pairs.csv")

# Map similarity type to numerical labels
similarity_mapping = {"direct": 1, "indirect": 0, "not-related": -1}
data["similarity_label"] = data["similarity_type"].map(similarity_mapping)

# Load weight vector
weight = np.load("weight_vector.npy")

# Function to get word vector from word2vec
def get_word_vector(word, word_model):
    return word_model[word] if word in word_model else np.zeros(word_model.vector_size)

# Prepare features using PCA
def prepare_features(df, word_model, weight):
    X, y = [], []
    for _, row in df.iterrows():
        word1_vector = get_word_vector(row["word1"], word_model)
        word2_vector = get_word_vector(row["word2"], word_model)
        
        if np.any(word1_vector) and np.any(word2_vector):
            weighted_word1 = word1_vector * weight
            weighted_word2 = word2_vector * weight

            dot_product = np.dot(weighted_word1, weighted_word2)
            norm_product = np.linalg.norm(weighted_word1) * np.linalg.norm(weighted_word2)
            similarity_score = dot_product / norm_product if norm_product != 0 else 0

            # Concatenate the word vectors
            X.append(np.concatenate([weighted_word1, weighted_word2]))
            y.append(row["similarity_label"])

    return np.array(X), np.array(y)

# Prepare the dataset
X, y = prepare_features(data, word_model, weight)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=100)  # Reduce to top 100 features
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Define classifiers and hyperparameters
classifiers = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "RFC": RandomForestClassifier()
}

param_grid = {
    "KNN": {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    "RFC": {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
}

# Perform RandomizedSearchCV for each classifier
best_models = {}
f1_scores = {}

for clf_name, clf in classifiers.items():
    search = RandomizedSearchCV(clf, param_grid[clf_name], n_iter=10, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)
    best_models[clf_name] = search.best_estimator_
    print(f"Best parameters for {clf_name}: {search.best_params_}")
    
    # Evaluate the model
    y_pred = search.best_estimator_.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores[clf_name] = f1
    print(f"F1 score for {clf_name}: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Similar", "Indirectly Similar", "Directly Similar"],
                yticklabels=["Not Similar", "Indirectly Similar", "Directly Similar"])
    plt.title(f"Confusion Matrix for {clf_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Select the best model based on F1 score
best_model_name = max(f1_scores, key=f1_scores.get)
best_model = best_models[best_model_name]
print(f"Best model: {best_model_name} with F1 score: {f1_scores[best_model_name]:.4f}")

# Final evaluation on the test set with the best model
if best_model_name == "KNN":
    final_model = KNeighborsClassifier(**best_model_params)
elif best_model_name == "SVM":
    final_model = SVC(**best_model_params)
elif best_model_name == "RFC":
    final_model = RandomForestClassifier(**best_model_params)

scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='f1_weighted')

print("Cross-Validation F1 Scores:", scores)
print("Mean F1 Score:", scores.mean())
print("Standard Deviation of F1 Score:", scores.std())

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Similar", "Indirectly Similar", "Directly Similar"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Similar", "Indirectly Similar", "Directly Similar"],
            yticklabels=["Not Similar", "Indirectly Similar", "Directly Similar"])
plt.title(f"Confusion Matrix for the Best Model ({best_model_name})")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
