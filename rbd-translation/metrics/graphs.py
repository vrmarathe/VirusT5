import csv
from Bio.Seq import Seq
import matplotlib.pyplot as plt



def count_differences_modified(seq1,seq2):
    differences = 0
    old_count=0
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    #print("\n Sequence 1 :",seq1,"Type:",type(seq1))
    #print("\n Sequence 2:",seq2,"Type:",type(seq2))
    for i, (base1, base2) in enumerate(zip(seq1, seq2), start=0):
        if(base1!=base2):
            #print("\n Base 1:",base1,"\n Base 2:",base2)
            try:
                old_count=count_position_dict[i]

            except KeyError as e:
                old_count=0
            #print("\nOld Count:",old_count)
            old_count=old_count+1
            #print("\nNew count:",old_count)
            count_position_dict.update({i:old_count})
            count_position_dict[i]= old_count
            #print("\n Count Dict ",count_position_dict)



    
count_position_dict={}
# Read sequences from the CSV file
input_file = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/metrics/comparision/aligned_sequences.csv"
#ref_sequence=ref_genome.seq
#print(ref_sequence)
#pool = multiprocessing.Pool()
#pool = multiprocessing.Pool(processes=4)

from Bio import SeqIO

with open("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/metrics/comparision/GISAID_ref_genome_RBD_nucleotides.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ref_sequence=record
        break



with open(input_file, "r") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        
        
        label_sequence = row["Labels"]
        #predicted_sequence = row["Predictions"]

        # Create Biopython Seq objects
        label_seq = Seq(label_sequence)
        #predicted_seq = Seq(predicted_sequence)

        # Count differences and get positions for this pair of sequences
        count_differences_modified(ref_sequence, label_seq)

        # # Append differences and positions to the lists
        # all_differences.append(differences)
        # all_positions.append(positions)





# print(all_differences[0:2])
# print(all_positions[0:2])
print("\n Processing DONE :",count_position_dict)

print("\n GENERATING THE OUTPUT FILE")
output_file = "output_label.csv"

# Open the CSV file for writing
with open(output_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(["Position", "Count"])

    # Write the data from the dictionary to the CSV file
    for key, value in count_position_dict.items():
        csv_writer.writerow([key, value])
        
        
count_position_dict={}

with open(input_file, "r") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        
        
        #label_sequence = row["Labels"]
        predicted_sequence = row["Predictions"]

        # Create Biopython Seq objects
        #label_seq = Seq(label_sequence)
        predicted_seq = Seq(predicted_sequence)

        # Count differences and get positions for this pair of sequences
        count_differences_modified(ref_sequence, predicted_seq)

        # # Append differences and positions to the lists
        # all_differences.append(differences)
        # all_positions.append(positions)

output_file = "output_pred.csv"

# Open the CSV file for writing
with open(output_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(["Position", "Count"])

    # Write the data from the dictionary to the CSV file
    for key, value in count_position_dict.items():
        csv_writer.writerow([key, value])

