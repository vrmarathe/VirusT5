# convert nucleotide to protein in csv

import pandas as pd
from Bio.Seq import Seq

# Define the translation function
def translate_nucleotides_to_protein(nucleotide_sequence):
    try:
        return str(Seq(nucleotide_sequence).translate())
    except Exception as e:
        print(f"Error translating sequence: {nucleotide_sequence}, Error: {e}")
        return None

# Load the input CSV file
input_csv_file = "df_predictions.csv"
df = pd.read_csv(input_csv_file, dtype=str)

# Initialize lists to hold the translated sequences
labels_prot = []
predictions_prot = []

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    label_nucleotide = row['Labels']
    prediction_nucleotide = row['Predictions']
    
    # Translate nucleotide sequences to protein sequences
    label_protein = translate_nucleotides_to_protein(label_nucleotide)
    prediction_protein = translate_nucleotides_to_protein(prediction_nucleotide)
    
    # Append the translated sequences to the lists
    labels_prot.append(label_protein)
    predictions_prot.append(prediction_protein)

# Add the new columns to the DataFrame
df['Labels_prot'] = labels_prot
df['Predictions_prot'] = predictions_prot

# Define the output CSV file
output_csv_file = "df_predictions_wt_prot.csv"

# Write the DataFrame to the new CSV file
df.to_csv(output_csv_file, index=False)

print(f"Output CSV file saved as {output_csv_file}")
