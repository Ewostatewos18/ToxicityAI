import pandas as pd # Not directly used in predict_logic.py for DataFrame operations, but often included
import re
import nltk
from nltk.corpus import stopwords # Still import, but 'stopwords' set itself won't be used for filtering here.
from nltk.stem import WordNetLemmatizer
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib # For loading the trained scikit-learn models
import os

# --- Configuration ---
MODEL_DIR = './models/' # Directory where trained models are saved
# Define the toxicity labels (must match the order used during training)
TOXICITY_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# --- NLTK component Initialization (Ensure these are downloaded from 02_preprocessing.py) ---
# If you run this script standalone and get NLTK LookupErrors, uncomment these:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english')) # No longer needed as we are removing stopword filtering

# --- 1. Re-define the Preprocessing Function (MUST BE IDENTICAL TO 02_preprocessing.py) ---
# This version performs essential cleaning but DOES NOT remove stopwords.
# This is generally preferred for Transformer-based models like DistilBERT.
def preprocess_text(text):
    # Ensure text is treated as string, handle potential non-string inputs gracefully
    if not isinstance(text, str):
        text = str(text) 
        if not text.strip(): # If it converts to an empty or whitespace-only string, return empty
            return ""

    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters (keep only letters, spaces, and underscores)
    text = re.sub(r'[^\w\s]', '', text) 
    
    # Remove extra whitespaces
    text = text.strip()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # --- CHANGE: Only lemmatize, DO NOT remove stopwords ---
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    return " ".join(processed_tokens)

# --- 2. Load Pre-trained DistilBERT Tokenizer and Model (for embeddings) ---
print("--- Loading DistilBERT Tokenizer and Model ---")
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval() # Set model to evaluation mode
    device = torch.device('cpu') # Predictions typically on CPU for a single inference, or 'mps'/'cuda' if available
    model.to(device)
    print(f"Using device for embeddings: {device}")
except Exception as e:
    print(f"Error loading DistilBERT model: {e}")
    print("Please ensure you have an internet connection for initial download or that the model is cached.")
    exit()

# --- 3. Load Trained Logistic Regression Models ---
print("\n--- Loading trained Logistic Regression models ---")
trained_lr_models = {}
for label in TOXICITY_LABELS:
    model_filename = os.path.join(MODEL_DIR, f'logistic_regression_{label}.joblib')
    try:
        lr_model = joblib.load(model_filename)
        trained_lr_models[label] = lr_model
        print(f"Loaded model for: '{label}'")
    except FileNotFoundError:
        print(f"Error: Model for '{label}' not found at {model_filename}.")
        print("Please ensure '04_train_model.py' was run successfully and saved all models.")
        exit()

# --- 4. Define Prediction Function ---
def predict_toxicity(comment):
    # Preprocess the comment
    cleaned_comment = preprocess_text(comment)
    
    if not cleaned_comment: # If comment becomes empty after preprocessing
        return {label: 0 for label in TOXICITY_LABELS}, "Comment became empty after preprocessing."

    # Generate embedding
    with torch.no_grad(): # Disable gradient calculations for inference
        encoded_input = tokenizer(
            cleaned_comment,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Take the [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy() # Convert to NumPy for scikit-learn
    
    # Make predictions using each trained Logistic Regression model
    predictions = {}
    for label in TOXICITY_LABELS:
        lr_model = trained_lr_models[label]
        # predict_proba returns probabilities for [class 0, class 1]
        # We want the probability of class 1 (toxic)
        probability = lr_model.predict_proba(embedding)[0][1]
        predictions[label] = probability # Store probability

    return predictions, "Success"

# --- 5. Interactive Prediction Loop ---
print("\n--- Toxicity Prediction AI Ready! ---")
print("Enter comments to predict toxicity. Type 'exit' to quit.")

while True:
    user_input = input("\nEnter comment: ")
    if user_input.lower() == 'exit':
        print("Exiting prediction AI. Goodbye!")
        break
    
    # Get predictions
    pred_probs, status = predict_toxicity(user_input)

    if status == "Success":
        print("\n--- Prediction Results ---")
        for label, prob in pred_probs.items():
            # A common threshold is 0.5, but you might adjust this based on desired sensitivity
            is_toxic = "YES" if prob >= 0.5 else "NO"
            print(f"  {label.capitalize()}: {is_toxic} (Probability: {prob:.4f})")
    else:
        print(f"\nPrediction failed: {status}")

print("\n--- Prediction script finished. ---")