import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK Downloads (Run only once!) ---
# If you haven't downloaded these NLTK data packages before, uncomment and run:
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt') # Required for word_tokenize

# --- Configuration ---
DATA_PATH = './data/train.csv' # Path to your dataset

# --- 1. Load the Data ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset '{DATA_PATH}' loaded successfully for preprocessing!")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure it's in the correct 'data' folder.")
    exit()

# --- 2. Initialize NLTK components ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 3. Define the Preprocessing Function ---
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters (keep only letters, spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespaces
    text = text.strip()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words and lemmatize
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    # Join tokens back into a string
    return " ".join(processed_tokens)

# --- 4. Apply Preprocessing to 'comment_text' column ---
print("\n--- Starting Text Preprocessing (This may take a few minutes for a large dataset) ---")
# Apply the function. Using .apply() is convenient but can be slow for very large datasets.
# For 160k rows, it should be manageable.
df['cleaned_comment_text'] = df['comment_text'].apply(preprocess_text)
print("--- Text Preprocessing Complete! ---")

# --- 5. Display Results ---
print("\n--- Original vs. Cleaned Comment Text (First 5 examples) ---")
for i in range(5):
    print(f"\nOriginal: {df['comment_text'][i]}")
    print(f"Cleaned: {df['cleaned_comment_text'][i]}")

print(f"\nDataFrame shape after preprocessing: {df.shape}")
print(f"New column 'cleaned_comment_text' created.")

# --- Save the Preprocessed Data ---
# It's good practice to save the preprocessed data so you don't have to re-run
# the preprocessing every time.
OUTPUT_PATH = './data/train_preprocessed.csv'
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nPreprocessed data saved to: {OUTPUT_PATH}")

print("\n--- Preprocessing script finished. ---")