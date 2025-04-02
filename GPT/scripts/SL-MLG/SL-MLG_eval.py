import os
import pandas as pd
import re

input_folder = 'input_folder'  # Replace with your input folder path
output_folder = 'output_folder'  # Replace with your output folder path

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to normalize text: removes extra spaces and converts to lowercase
def normalize_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()  # Convert to lowercase for case-insensitive matching

# List of predefined categories 
predefined_categories = [
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

# Normalize the predefined categories for consistent matching
normalized_categories = [normalize_text(cat) for cat in predefined_categories]

# Function to extract categories from the Ground Truth string
def extract_categories(ground_truth_str):
    extracted_categories = []
    # Normalize the Ground Truth string
    ground_truth_normalized = normalize_text(ground_truth_str)
    for category in normalized_categories:
        # Use regex to find exact matches of the category in the Ground Truth string
        pattern = re.escape(category)
        if re.search(pattern, ground_truth_normalized):
            extracted_categories.append(category)
    return extracted_categories

# Get a list of CSV files in the input folder
csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

# Process each CSV file
for csv_file in csv_files:
    input_file_path = os.path.join(input_folder, csv_file)
    df = pd.read_csv(input_file_path)

    # Initialize a list to store scores
    scores = []

    for index, row in df.iterrows():
        # Get the Ground Truth and LLM Output
        ground_truth = str(row['Ground Truth'])
        llm_output = str(row['LLM Output'])

        # Normalize the LLM Output
        llm_output_normalized = normalize_text(llm_output)

        # Extract categories from Ground Truth
        categories = extract_categories(ground_truth)

        # Normalize each category in Ground Truth
        categories_normalized = [normalize_text(cat) for cat in categories]

        # Check if the normalized LLM Output matches any of the normalized Ground Truth categories
        if llm_output_normalized in categories_normalized:
            score = 1
        else:
            score = 0

        # Append the score to the list
        scores.append(score)

    # Add the scores to the DataFrame
    df['Score'] = scores

    # Calculate the average score
    average_score = sum(scores) / len(scores) if scores else 0

    # Insert a row at the top with the average score
    average_row = pd.DataFrame([{'Ground Truth': 'Average', 'LLM Output': '', 'Score': average_score}])
    df = pd.concat([average_row, df], ignore_index=True)

    # Save the updated DataFrame to the output folder
    output_file_path = os.path.join(output_folder, csv_file)
    df.to_csv(output_file_path, index=False)

    print(f'Processed {csv_file} and saved to {output_file_path}')
