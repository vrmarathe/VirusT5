from Bio import Align
import pandas as pd
from multiprocessing import Pool

# Load the DataFrame from the CSV file
#df = pd.read_csv("df_predictions.csv")
df = pd.read_csv("df_predictions_wt_prot.csv")
# Create a PairwiseAligner with the desired parameters
aligner = Align.PairwiseAligner()
aligner.mode = 'global'  # Use global alignment mode (other modes: local, semiglobal)
aligner.open_gap_score = -5  # Adjust this value as needed
aligner.extend_gap_score = -1  # Adjust this value as needed

# Function to perform alignment and calculate scores
def align_and_calculate_score(args):
    index, row = args
    seq1 = row["Labels_prot"]
    seq2 = row["Predictions_prot"]

    try:
        # Perform pairwise alignment
        alignments = aligner.align(seq1, seq2)

        if alignments:
            # Get the first (best) alignment
            alignment = alignments[0]

            # Calculate the alignment score
            alignment_score = alignment.score
            c=alignment.counts()
            identity_percentage=c.identities/(c.gaps+c.identities+c.mismatches)
            # Calculate the percentage of identity
            #sequence_length = max(len(seq1), len(seq2))
            #identity_percentage = (alignment_score / sequence_length) * 100

            return alignment_score, identity_percentage
        else:
            # No alignment found for this pair of sequences
            return None, None
    except OverflowError:
        # Handle OverflowError and return None
        return None, None

# Create a list of arguments for the function
args_list = [(index, row) for index, row in df.iterrows()]

# Create a multiprocessing pool
num_processes = 62  # Adjust as needed
with Pool(processes=num_processes) as pool:
    results = pool.map(align_and_calculate_score, args_list)

# Extract alignment scores and identity percentages from the results
alignment_scores, identity_percentages = zip(*results)

# Add new columns for alignment scores and identity percentages to the DataFrame
df["Alignment_Score"] = alignment_scores
df["Identity_Percentage"] = identity_percentages

# Save the updated DataFrame to a new CSV file
df.to_csv("aligned_sequences_prot.csv", index=False)
