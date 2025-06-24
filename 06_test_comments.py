import sys
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- IMPORTANT: Ensure predict_logic.py is correctly structured ---
# This line assumes your renamed 05_predict.py (now predict_logic.py)
# contains a function named `predict_comment` that takes a string and
# returns a dictionary of predictions like:
# {
#     'toxic': {'prediction': 'YES'/'NO', 'probability': float},
#     'severe_toxic': {'prediction': 'YES'/'NO', 'probability': float},
#     # ... for all 6 categories
# }

# Add the current directory to sys.path to allow importing local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from predict_logic import predict_comment
except ImportError:
    print("Error: Could not import 'predict_comment' from 'predict_logic.py'.")
    print("Please ensure you have renamed 05_predict.py to predict_logic.py")
    print("AND that predict_logic.py contains a function named 'predict_comment'.")
    sys.exit(1)

# --- Your Test Comments ---
# Replace these example comments with your actual 200 comments.
# Format: (comment_text, is_truly_toxic)
# 'is_truly_toxic': 1 for bad/toxic comments, 0 for good/non-toxic comments.
# Make sure you have approx. 150 comments with 1 and 50 comments with 0.
test_comments_with_true_labels = [
    ("You are an idiot and your opinion is worthless.", 1),
    ("That's an interesting point of view.", 0),
    ("I'm going to find you and make you regret that.", 1),
    ("Thank you for sharing your thoughts.", 0),
    ("What the hell are you talking about?", 1),
    ("I agree with your assessment.", 0),
    ("Your comments are disgusting and hateful.", 1),
    ("The cat jumped over the fence.", 0),
    ("This is such a ridiculous and pathetic argument.", 1),
    ("I hope you have a wonderful day!", 0),
    ("You deserve to suffer for what you've done.", 1),
    ("Can we discuss this calmly?", 0),
    ("Seriously, you're a complete waste of space.", 1),
    ("Learning new things is always exciting.", 0),
    ("I'll make sure you regret every single word you've typed.", 1),
    ("Hope to see you soon.", 0),
    ("Your actions are so utterly repulsive.", 1),
    ("The sky is blue today.", 0),
    ("You are a piece of trash.", 1),
    ("Have a great evening!", 0),
    # Add your remaining 180 comments here...
    # Example for toxic: ("Go burn in hell, you piece of garbage.", 1),
    # Example for non-toxic: ("This project looks very promising.", 0),
]

# --- Initialize Lists for Evaluation ---
true_labels_toxic = []
predicted_labels_toxic = []
# You can extend this for other categories if your boss wants detailed per-category metrics.
# e.g., true_labels_insult = []
#       predicted_labels_insult = []

print("\n--- Running Custom Test on General Comments ---\n")

# --- Run Predictions and Collect Results ---
for i, (comment, true_is_toxic) in enumerate(test_comments_with_true_labels):
    print(f"Testing comment {i+1}/{len(test_comments_with_true_labels)}: '{comment}'")
    
    try:
        results = predict_comment(comment) # Call the prediction function
    except Exception as e:
        print(f"  Error predicting for comment '{comment}': {e}")
        # If an error occurs, assume a default (e.g., non-toxic) or skip this comment
        is_predicted_toxic = 0 
        print(f"  Skipping this comment due to prediction error.\n")
        continue

    # Extract overall toxic prediction for evaluation
    # This assumes 'toxic' is one of the keys returned by predict_comment
    if 'toxic' in results and 'prediction' in results['toxic']:
        is_predicted_toxic = 1 if results['toxic']['prediction'] == 'YES' else 0
    else:
        print(f"  Warning: 'toxic' prediction not found in results for '{comment}'. Assuming NO.\n")
        is_predicted_toxic = 0 
    
    true_labels_toxic.append(true_is_toxic)
    predicted_labels_toxic.append(is_predicted_toxic)

    # Print results for the 'toxic' category for easy review
    print(f"  AI Predicted Toxic: {results.get('toxic', {}).get('prediction', 'N/A')} (Prob: {results.get('toxic', {}).get('probability', 0.0):.4f})")
    print(f"  True Label (Toxic): {'YES' if true_is_toxic == 1 else 'NO'}\n")

# --- Evaluate Overall 'Toxic' Category Performance ---
print("\n--- Overall 'Toxic' Category Performance ---")

if len(true_labels_toxic) > 0:
    accuracy = accuracy_score(true_labels_toxic, predicted_labels_toxic)
    precision = precision_score(true_labels_toxic, predicted_labels_toxic, zero_division=0)
    recall = recall_score(true_labels_toxic, predicted_labels_toxic, zero_division=0)
    f1 = f1_score(true_labels_toxic, predicted_labels_toxic, zero_division=0)

    print(f"Accuracy (Overall): {accuracy:.4f}")
    print(f"Precision (Toxic): {precision:.4f}")
    print(f"Recall (Toxic): {recall:.4f}")
    print(f"F1-Score (Toxic): {f1:.4f}")
else:
    print("No comments processed for evaluation.")

print("\n--- Test Complete ---")