import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score
import joblib # For saving/loading models

# --- Configuration ---
PREPROCESSED_DATA_PATH = './data/train_preprocessed.csv'
EMBEDDINGS_PATH = './data/comment_embeddings.pt'
MODEL_OUTPUT_DIR = './models/' # Directory to save trained models

# Ensure the models directory exists
import os
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Define the toxicity labels (these are your target columns)
TOXICITY_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- 1. Load Data ---
print("--- Loading preprocessed data and embeddings ---")
try:
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data loaded from: {PREPROCESSED_DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Preprocessed dataset not found at {PREPROCESSED_DATA_PATH}.")
    exit()

# Align DataFrame with embeddings by dropping rows with NaN in cleaned_comment_text
# This step was done in 03_feature_engineering.py before generating embeddings.
df.dropna(subset=['cleaned_comment_text'], inplace=True)
print(f"DataFrame shape after aligning with embeddings (dropping NaNs): {df.shape}")

try:
    comment_embeddings = torch.load(EMBEDDINGS_PATH)
    # Convert PyTorch tensor to NumPy array for scikit-learn
    X = comment_embeddings.numpy()
    print(f"Embeddings loaded from: {EMBEDDINGS_PATH}")
    print(f"Shape of features (X): {X.shape}")
except FileNotFoundError:
    print(f"Error: Embeddings file not found at {EMBEDDINGS_PATH}.")
    print("Please ensure '03_feature_engineering.py' was run successfully and saved the embeddings.")
    exit()

# Ensure the number of samples matches between features (X) and labels (df)
if X.shape[0] != df.shape[0]:
    print(f"Warning: Mismatch in number of samples! Features: {X.shape[0]}, Labels: {df.shape[0]}")
    # This might happen if '03_feature_engineering.py' skipped some rows due to NaN comments.
    # We will align them by using the 'cleaned_comment_text' column from the df that was used to generate embeddings.
    # Note: df.dropna(subset=['cleaned_comment_text'], inplace=True) was done in 03_feature_engineering.py
    # If the embeddings were generated *after* dropping NaNs, then the shapes should match.
    # For robustness, we should re-align them if they somehow differ.
    # Assuming df used to generate embeddings is the one after dropping NaNs in 03_feature_engineering.py
    # If not, you might need to load the *original* df here, and then drop NaNs
    # to make sure its size matches the embeddings you saved.
    # For now, let's assume `X` corresponds directly to `df` *after* dropping NaNs in 03_feature_engineering.py.
    # We will confirm by printing shapes.
    pass # If they don't match, you'll need to figure out which rows were excluded
        # For simplicity, we'll proceed assuming they align.

Y = df[TOXICITY_LABELS].values # Get the labels as a NumPy array
print(f"Shape of labels (Y): {Y.shape}")

# --- 2. Split Data (Training, Validation, Test Sets) ---
# First split: train (80%) and temp (20%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
# Second split: validation (10% of total), test (10% of total) from temp
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"\nData split sizes:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Validation set: {X_val.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")


# --- 3. Train a Binary Classifier for Each Label ---
print("\n--- Training models for each toxicity label ---")
trained_models = {}
for i, label in enumerate(TOXICITY_LABELS):
    print(f"\nTraining model for: '{label}'...")
    # Initialize Logistic Regression model
    # solver='liblinear' is good for small datasets and binary classification
    # class_weight='balanced' helps with imbalanced classes (common in toxicity data)
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42, max_iter=1000)

    # Train the model for the current label
    model.fit(X_train, Y_train[:, i]) # Y_train[:, i] selects the column for the current label

    # Store the trained model
    trained_models[label] = model

    # --- 4. Evaluate on Validation Set ---
    Y_pred_val = model.predict(X_val)
    Y_prob_val = model.predict_proba(X_val)[:, 1] # Probability of the positive class (1)

    print(f"--- Evaluation for '{label}' on Validation Set ---")
    print(classification_report(Y_val[:, i], Y_pred_val, zero_division=0)) # zero_division=0 to avoid warning if no true instances of a class
    print(f"F1-Score: {f1_score(Y_val[:, i], Y_pred_val, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(Y_val[:, i], Y_prob_val):.4f}")
    print(f"Accuracy: {accuracy_score(Y_val[:, i], Y_pred_val):.4f}")

    # --- 5. Save the Trained Model ---
    model_filename = os.path.join(MODEL_OUTPUT_DIR, f'logistic_regression_{label}.joblib')
    joblib.dump(model, model_filename)
    print(f"Model for '{label}' saved to: {model_filename}")

print("\n--- All models trained and saved. ---")

# --- Final Evaluation on Test Set (Optional, but good practice) ---
# Only evaluate on the test set once, after model selection and hyperparameter tuning (which we're skipping for simplicity)
print("\n--- Final Evaluation on Test Set ---")
overall_f1_scores = []
for i, label in enumerate(TOXICITY_LABELS):
    model = trained_models[label] # Get the trained model
    Y_pred_test = model.predict(X_test)
    Y_prob_test = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(Y_test[:, i], Y_pred_test, zero_division=0)
    roc_auc = roc_auc_score(Y_test[:, i], Y_prob_test)
    accuracy = accuracy_score(Y_test[:, i], Y_pred_test)

    print(f"\n--- Test Results for '{label}' ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    overall_f1_scores.append(f1)

print(f"\nAverage F1-score across all labels (Test Set): {np.mean(overall_f1_scores):.4f}")
print("\n--- Model training and evaluation script finished. ---")