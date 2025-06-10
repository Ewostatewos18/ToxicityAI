import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # This library helps us work with file paths

# --- Configuration ---
# Define the path to your dataset folder
# We use os.path.join to make sure it works correctly on Windows, macOS, and Linux
DATA_FOLDER = os.path.join('.', 'data') # This means 'current directory' + 'data' folder
TRAIN_FILE = os.path.join(DATA_FOLDER, 'train.csv') # Path to the train.csv file

# Define the toxicity labels as they appear in the dataset columns
TOXICITY_LABELS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]

print("--- Starting Data Loading and Exploration ---")
print(f"Looking for dataset at: {TRAIN_FILE}")

# --- Load the Dataset ---
try:
    df_train = pd.read_csv(TRAIN_FILE)
    print("Dataset 'train.csv' loaded successfully!")
except FileNotFoundError:
    print(f"Error: '{TRAIN_FILE}' not found.")
    print("Please ensure 'train.csv' is in the 'data/' folder inside your project.")
    print("Download it from: https://www.kaggle.com/c/toxic-comment-classification-challenge/data")
    exit() # Stop the script if the file isn't found

# --- Explore the Data ---
print("\n--- First 5 rows of the dataset ---")
# .head() shows the first few rows of the DataFrame
print(df_train.head())

print("\n--- Dataset Information (Columns, Non-Null Counts, Data Types) ---")
# .info() gives a summary of the DataFrame structure
df_train.info()

print("\n--- Column Names in the Dataset ---")
# .columns shows all the column names
print(df_train.columns.tolist()) # .tolist() converts it to a regular list for easier reading

print(f"\n--- Identifying Toxicity Labels: {TOXICITY_LABELS} ---")

print("\n--- Distribution of Toxicity Labels (Count of '1's in each category) ---")
# .sum() on the label columns counts how many times '1' appears (meaning it's toxic)
label_counts = df_train[TOXICITY_LABELS].sum()
print(label_counts)

# --- Visualize the Distribution ---
plt.figure(figsize=(10, 6)) # Set the size of the plot
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title('Distribution of Toxicity Categories') # Title of the plot
plt.xlabel('Toxicity Category') # Label for the horizontal axis
plt.ylabel('Number of Comments') # Label for the vertical axis
plt.xticks(rotation=45, ha='right') # Rotate category names for readability
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show() # Display the plot

print("\n--- Sample Comments for Each Toxicity Category ---")
# Display a few sample comments from each toxicity category
for label in TOXICITY_LABELS:
    # Filter comments where the current label is 1 (toxic)
    # .sample() picks random comments, min(3, ...) ensures we don't ask for more than available
    # .tolist() converts the selected comments to a Python list
    sample_comments = df_train[df_train[label] == 1]['comment_text'].sample(
        min(3, df_train[df_train[label] == 1].shape[0]), random_state=42
    ).tolist()

    if sample_comments: # Check if there are any comments for this label
        print(f"\n--- Sample '{label}' comments ---")
        for i, comment in enumerate(sample_comments):
            # Print only the first 200 characters for brevity
            print(f"  {i+1}. {comment[:200]}...")
    else:
        print(f"  No comments found for '{label}' in samples.")

print("\n--- Data Loading and Exploration Complete ---")