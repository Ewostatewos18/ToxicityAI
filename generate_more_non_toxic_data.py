import random
import csv
import pandas as pd # Import pandas to find the last ID

def introduce_typos(word, num_typos=1):
    """Introduces a specified number of random typos (deletion, insertion, substitution, transposition)."""
    word_list = list(word)
    word_len = len(word_list)
    
    if word_len == 0:
        return word

    for _ in range(num_typos):
        typo_type = random.choice(['delete', 'insert', 'substitute', 'transpose'])
        
        if typo_type == 'delete':
            if word_len > 1:
                idx = random.randint(0, word_len - 1)
                del word_list[idx]
                word_len -= 1
        elif typo_type == 'insert':
            idx = random.randint(0, word_len)
            random_char = chr(random.randint(97, 122)) # random lowercase letter
            word_list.insert(idx, random_char)
            word_len += 1
        elif typo_type == 'substitute':
            if word_len > 0:
                idx = random.randint(0, word_len - 1)
                random_char = chr(random.randint(97, 122))
                word_list[idx] = random_char
        elif typo_type == 'transpose':
            if word_len > 1:
                idx1 = random.randint(0, word_len - 2)
                idx2 = idx1 + 1
                word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
    
    return "".join(word_list)

# --- Configuration ---
NEW_NON_TOXIC_CSV_PATH = './data/new_non_toxic_comments.csv'

# Common non-toxic words to generate misspellings for
positive_words = [
    "good", "great", "excellent", "wonderful", "amazing", "happy", "love", "awesome",
    "beautiful", "nice", "fantastic", "hello", "hi", "thank you", "please", "sorry",
    "yes", "no", "okay", "fine", "cool", "super", "best", "kind", "friendly",
    "helpful", "perfect", "brilliant", "terrific", "lovely", "peace", "joy", "hope",
    "faith", "smile", "laugh", "friend", "family", "fun", "game", "book", "movie",
    "music", "art", "work", "day", "night", "morning", "evening", "time", "place",
    "home", "city", "country", "agree", "understand", "welcome", "glad", "blessed"
]

# --- Find the last ID in the existing CSV ---
try:
    existing_df = pd.read_csv(NEW_NON_TOXIC_CSV_PATH)
    # Extract number from the last ID (e.g., 'new_0210' -> 210)
    last_id_num = int(existing_df['id'].iloc[-1].replace('new_', ''))
    next_id_num = last_id_num + 1
except (FileNotFoundError, IndexError):
    # If file doesn't exist or is empty, start from 1
    next_id_num = 1
print(f"Starting new IDs from: new_{next_id_num:04d}")

# --- Generate new data ---
new_rows = []
for word in positive_words:
    # Add the original word as a non-toxic example
    new_rows.append([f"new_{next_id_num:04d}", word, 0, 0, 0, 0, 0, 0])
    next_id_num += 1

    # Add misspellings of the word
    for _ in range(5): # Generate 5 misspellings per word
        misspelled_word = introduce_typos(word, num_typos=random.randint(1, 2)) # 1 or 2 typos
        if misspelled_word and misspelled_word != word: # Avoid empty strings and duplicates of original word
            new_rows.append([f"new_{next_id_num:04d}", misspelled_word, 0, 0, 0, 0, 0, 0])
            next_id_num += 1

    # Also add some misspellings within simple, non-toxic phrases
    # This helps contextual learning
    phrases = [
        "this is {word}", "that's a {word} idea", "feeling {word} today",
        "very {word}", "it was so {word}", "you look {word}", "{word} job"
    ]
    for phrase_template in phrases:
        misspelled_word = introduce_typos(word, num_typos=random.randint(1, 2))
        if misspelled_word and misspelled_word != word:
            new_rows.append([f"new_{next_id_num:04d}", phrase_template.format(word=misspelled_word), 0, 0, 0, 0, 0, 0])
            next_id_num += 1


# --- Append new data to the CSV ---
# Use 'a' mode for append, newline='' to prevent extra blank rows
with open(NEW_NON_TOXIC_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in new_rows:
        writer.writerow(row)

print(f"\nSuccessfully appended {len(new_rows)} new non-toxic entries to '{NEW_NON_TOXIC_CSV_PATH}'.")
print("Remember to re-run the full ML pipeline now!")