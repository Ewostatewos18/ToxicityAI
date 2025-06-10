import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm # For progress bar

# --- Configuration ---
PREPROCESSED_DATA_PATH = './data/train_preprocessed.csv'
EMBEDDINGS_OUTPUT_PATH = './data/comment_embeddings.pt' # PyTorch tensor file
BATCH_SIZE = 32 # Adjust based on your Mac's RAM and CPU/GPU (higher is faster if memory allows)

# --- 1. Load the Preprocessed Data ---
try:
    df = pd.read_csv(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed dataset '{PREPROCESSED_DATA_PATH}' loaded successfully!")
    print(f"DataFrame shape: {df.shape}")
    print(f"First 5 cleaned comments:\n{df['cleaned_comment_text'].head()}")
except FileNotFoundError:
    print(f"Error: Preprocessed dataset not found at {PREPROCESSED_DATA_PATH}.")
    print("Please ensure '02_preprocessing.py' was run successfully and saved the file.")
    exit()

# Filter out any NaN values that might have resulted from preprocessing empty comments (though unlikely with current script)
df.dropna(subset=['cleaned_comment_text'], inplace=True)
print(f"DataFrame shape after dropping NaNs: {df.shape}")


# --- 2. Load Pre-trained DistilBERT Tokenizer and Model ---
print("\n--- Loading DistilBERT Tokenizer and Model ---")
# Use 'distilbert-base-uncased' for the uncased version
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Set model to evaluation mode (important for inference, disables dropout)
model.eval()

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# --- 3. Tokenize the cleaned comment text ---
print("\n--- Tokenizing comments (This may take a while) ---")
# `truncation=True` truncates comments longer than model's max input length (512 tokens for DistilBERT)
# `padding=True` pads shorter comments to the max length
# `return_tensors='pt'` returns PyTorch tensors
encoded_input = tokenizer(
    df['cleaned_comment_text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt'
)

input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

# Create a TensorDataset and DataLoader for efficient batching
dataset = TensorDataset(input_ids, attention_mask)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# --- 4. Generate Embeddings ---
print("\n--- Generating comment embeddings (This will take significant time) ---")
all_embeddings = []
# Disable gradient calculations for inference to save memory and speed up
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Generating Embeddings"):
        batch_input_ids, batch_attention_mask = batch
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        
        # We take the [CLS] token embedding, which is the first token's embedding (outputs.last_hidden_state[:, 0, :])
        # This is commonly used as a fixed-size representation of the entire sequence.
        embeddings = outputs.last_hidden_state[:, 0, :].cpu() # Move to CPU for storage
        all_embeddings.append(embeddings)

# Concatenate all embeddings into a single tensor
comment_embeddings = torch.cat(all_embeddings, dim=0)

print(f"\n--- Embeddings Generation Complete! ---")
print(f"Shape of generated embeddings: {comment_embeddings.shape}") # Expected: (num_samples, embedding_dim, e.g., 768)

# --- 5. Save the Embeddings ---
torch.save(comment_embeddings, EMBEDDINGS_OUTPUT_PATH)
print(f"Comment embeddings saved to: {EMBEDDINGS_OUTPUT_PATH}")

print("\n--- Feature Engineering script finished. ---")