import pandas as pd
import re
import nltk
from nltk.corpus import stopwords # Import still needed if you use other NLTK features, but 'stopwords' set itself won't be used for filtering here.
from nltk.stem import WordNetLemmatizer

# --- NLTK Downloads (Run only once if you haven't downloaded these before!) ---
# If you get NLTK LookupError, uncomment the lines below and run this script.
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt') # Required for word_tokenize

# --- Configuration ---
DATA_PATH = './data/train.csv' # Path to your dataset
OUTPUT_PATH = './data/train_preprocessed.csv' # Path to save the preprocessed data

# --- 1. Load the Data ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset '{DATA_PATH}' loaded successfully for preprocessing!")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure it's in the correct 'data' folder.")
    exit()

# --- 2. Initialize NLTK components ---
lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english')) # No longer needed as we are removing stopword filtering

# --- 3. Define the Preprocessing Function (General Solution) ---
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
    # This keeps characters like 'i', 'you', 'hey' as words.
    text = re.sub(r'[^\w\s]', '', text) 
    
    # Remove extra whitespaces
    text = text.strip()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # --- CHANGE: Only lemmatize, DO NOT remove stopwords ---
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    return " ".join(processed_tokens)

# --- 4. Apply Preprocessing to 'comment_text' column ---
print("\n--- Starting Text Preprocessing (This may take a few minutes for a large dataset) ---")
df['cleaned_comment_text'] = df['comment_text'].apply(preprocess_text)
print("--- Text Preprocessing Complete! ---")

# --- 5. Display Results ---
print("\n--- Original vs. Cleaned Comment Text (First 5 examples) ---")
# Filter out potential empty cleaned comments for display purposes if any occurred
display_df = df[df['cleaned_comment_text'].str.strip() != ''].head(5)
for i in range(len(display_df)):
    print(f"\nOriginal: {display_df['comment_text'].iloc[i]}")
    print(f"Cleaned: {display_df['cleaned_comment_text'].iloc[i]}")

print(f"\nDataFrame shape after preprocessing: {df.shape}")
print(f"New column 'cleaned_comment_text' created.")

# --- Save the Preprocessed Data ---
# It's good practice to save the preprocessed data so you don't have to re-run
# the preprocessing every time.
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nPreprocessed data saved to: {OUTPUT_PATH}")

print("\n--- Preprocessing script finished. ---")