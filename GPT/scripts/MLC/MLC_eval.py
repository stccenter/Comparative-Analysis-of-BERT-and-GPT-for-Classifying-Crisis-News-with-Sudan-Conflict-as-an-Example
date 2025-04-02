import os
import csv
import pandas as pd
import re

# Define the list of included categories
included_categories = [
    "Unlawful detention",
    "Human trafficking",
    "Enslavement",
    "Willful killing of civilians",
    "Mass execution",
    "Kidnapping",
    "Extrajudicial killing",
    "Forced disappearance",
    "Damage or destruction of civilian critical infrastructure",
    "Damage or destruction, looting, or theft of cultural heritage",
    "Military operations (battle, shelling)",
    "Gender-based or other conflict-related sexual violence",
    "Violent crackdowns on protesters/opponents/civil rights abuse",
    "Indiscriminate use of weapons",
    "Torture or indications of torture",
    "Persecution based on political, racial, ethnic, gender, or sexual orientation",
    "Movement of military, paramilitary, or other troops and equipment"
]

# Create a set for faster lookup
included_categories_set = set(included_categories)

def normalize_category(cat):
    # Normalize category for matching
    return cat.strip().lower()

def extract_categories(text, category_set):
    # Split by comma or newline
    splits = re.split(r',|\n', text)
    categories = []
    temp = ''
    for part in splits:
        part = part.strip()
        if temp:
            temp += ', ' + part
        else:
            temp = part
        normalized = normalize_category(temp)
        if normalized in (cat.lower() for cat in category_set):
            categories.append(temp)
            temp = ''
    if temp:
        normalized = normalize_category(temp.strip(', '))
        if normalized in (cat.lower() for cat in category_set):
            categories.append(temp)
    return set(categories)

def calculate_metrics(ground_truth, predicted):
    true_positives = ground_truth & predicted
    false_positives = predicted - ground_truth
    false_negatives = ground_truth - predicted
    
    if ground_truth == predicted:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        if len(predicted) == 0:
            precision = 0.0
        else:
            precision = len(true_positives) / (len(true_positives) + len(false_positives))
        if len(ground_truth) == 0:
            recall = 0.0
        else:
            recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1, true_positives, false_positives, false_negatives

# Paths
input_folder = 'input_folder'  # Replace with your input folder path
output_folder = 'output_folder'  # Replace with your output folder path

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename.replace('.csv', '_evaluated.csv'))
        category_summary_file = os.path.join(output_folder, filename.replace('.csv', '_category_summary.csv'))
        
        df = pd.read_csv(input_file)

        # Remove rows where 'Ground Truth' or 'LLM Output' is empty
        df = df.dropna(subset=['Ground Truth', 'LLM Output'])

        precisions = []
        recalls = []
        f1s = []
        accuracies = []  
        tp_list = []
        fp_list = []
        fn_list = []

        # Initialize per-category FP and FN counts
        category_fp_counts = {cat: 0 for cat in included_categories}
        category_fn_counts = {cat: 0 for cat in included_categories}

        for index, row in df.iterrows():
            gt_text = str(row['Ground Truth'])
            llm_text = str(row['LLM Output'])

            # Extract included categories from ground truth and LLM output
            ground_truth_categories = extract_categories(gt_text, included_categories_set)
            predicted_categories = extract_categories(llm_text, included_categories_set)

            # Calculate "exact match" accuracy (order doesn't matter because they're sets)
            accuracy = 1 if ground_truth_categories == predicted_categories else 0
            accuracies.append(accuracy)

            # Proceed with calculation of precision, recall, and F1
            precision, recall, f1, true_positives, false_positives, false_negatives = calculate_metrics(
                ground_truth_categories, predicted_categories
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            tp_list.append('; '.join(true_positives))
            fp_list.append('; '.join(false_positives))
            fn_list.append('; '.join(false_negatives))

            # Update per-category FP and FN counts
            for cat in false_positives:
                if cat in category_fp_counts:
                    category_fp_counts[cat] += 1
            for cat in false_negatives:
                if cat in category_fn_counts:
                    category_fn_counts[cat] += 1

        # Add metrics to the DataFrame
        df['Accuracy'] = accuracies
        df['Precision'] = precisions
        df['Recall'] = recalls
        df['F1'] = f1s
        df['True Positives'] = tp_list
        df['False Positives'] = fp_list
        df['False Negatives'] = fn_list

        # Calculate averages
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        # Create a DataFrame for the averages
        averages_df = pd.DataFrame({
            'Ground Truth': ['Average'],
            'LLM Output': [''],
            'Accuracy': [avg_accuracy],
            'Precision': [avg_precision],
            'Recall': [avg_recall],
            'F1': [avg_f1],
            'True Positives': [''],
            'False Positives': [''],
            'False Negatives': ['']
        })

        # Combine the averages row with the original DataFrame
        df = pd.concat([averages_df, df], ignore_index=True)

        # Save the new CSV
        df.to_csv(output_file, index=False)

        # Create per-category FP and FN counts DataFrame
        category_counts_df = pd.DataFrame({
            'Category': included_categories,
            'False Positives': [category_fp_counts[cat] for cat in included_categories],
            'False Negatives': [category_fn_counts[cat] for cat in included_categories]
        })

        # Save the per-category counts to a new CSV
        category_counts_df.to_csv(category_summary_file, index=False)

print("Evaluation complete. Evaluated files and category summaries are saved in the output folder.")
