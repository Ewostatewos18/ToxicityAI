# 01_5_merge_data.py (or integrate into 01_load_data.py)

import pandas as pd
import numpy as np # For potential NaN handling if IDs are missing

# --- Paths to your files ---
original_train_csv_path = 'data/train.csv' # Assuming train.csv is in a 'data' folder
new_comments_csv_path = 'data/new_non_toxic_comments.csv' # This is the new file you created
output_train_csv_path = 'data/train.csv' # Overwrite the original train.csv with combined data

print(f"Loading original training data from: {original_train_csv_path}")
try:
    df_original = pd.read_csv(original_train_csv_path)
    print(f"Original data shape: {df_original.shape}")
except FileNotFoundError:
    print(f"Error: Original train.csv not found at {original_train_csv_path}")
    exit() # Exit if the original file isn't found
except Exception as e:
    print(f"Error loading original train.csv: {e}")
    exit()

print(f"Loading new comments from: {new_comments_csv_path}")
try:
    df_new = pd.read_csv(new_comments_csv_path)
    print(f"New comments shape: {df_new.shape}")
except FileNotFoundError:
    print(f"Error: New comments CSV not found at {new_comments_csv_path}")
    exit() # Exit if the new file isn't found
except Exception as e:
    print(f"Error loading new comments CSV: {e}")
    exit()

# --- Generate unique IDs for new comments if they are not already unique ---
# This is important to avoid duplicate IDs if your original data has its own ID scheme
# Let's assume original IDs are strings or can be converted to strings to append safely
max_id_num = 0
if not df_original['id'].empty:
    # Try to find the max numerical part of existing IDs to ensure new ones are unique
    # This is a bit complex for mixed ID types, but for simplicity, let's just use string concat
    if df_original['id'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x)).any():
        max_id_num = df_original['id'].astype(str).str.extract(r'(\d+)').astype(float).max()
        if pd.isna(max_id_num): # Handle case where no numeric IDs are found
            max_id_num = 0
        else:
            max_id_num = int(max_id_num)


new_ids = [f"custom_{i + max_id_num + 1}" for i in range(len(df_new))]
df_new['id'] = new_ids


# Ensure all columns are present in df_new, even if they are just 0s
# This aligns columns before concatenation
missing_cols_in_new = set(df_original.columns) - set(df_new.columns)
for col in missing_cols_in_new:
    df_new[col] = 0 # Assign 0 to missing label columns

# Ensure column order is the same
df_new = df_new[df_original.columns]


print("Concatenating dataframes...")
df_combined = pd.concat([df_original, df_new], ignore_index=True)
print(f"Combined data shape: {df_combined.shape}")

print(f"Saving combined data to: {output_train_csv_path}")
df_combined.to_csv(output_train_csv_path, index=False)
print("Data combined and saved successfully!")