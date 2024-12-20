##IMPORTED LIBRARIES ##

from Bio import SeqIO
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import csv
import pandas as pd
## GETTING THE REFERNCE RBD SEQ

with open("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/metrics/comparision/GISAID_ref_genome_RBD_nucleotides.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ref_sequence=record
        break

print("\n\n Reference Record :",ref_sequence)
print("\n\n")
print("\n\nData Type:",type(ref_sequence))

#print("\n\n Converting to Text/String  !!")

ref_string=str(ref_sequence.seq)

print("\n\n Ref Seq:",ref_string,"\n\n Data type:",type(ref_string))


# tokenizer = AutoTokenizer.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_alpha/checkpoint-9500",load_best_model_at_end=True)

# print("\n\n TOKENIZER IMPORTED !!")

# model = AutoModelForSeq2SeqLM.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_alpha/checkpoint-9500")
# print("\n\n MODEL IMPORTED !!")


def preprocess_sequence(text_input):
    prefix = "translate Seq1 to Seq2: "
    return prefix+text_input

def generate_sequence(text,tokenizer,model):
    inputs = tokenizer(text, return_tensors="pt").input_ids
    #print("\n\n TOKENIZED INPUT !!")
    outputs = model.generate(inputs, max_new_tokens=667, do_sample=True, top_k=30, top_p=0.95)
    #print("\n\n Output Sequence : ",tokenizer.decode(outputs[0], skip_special_tokens=True))
    return str.upper(tokenizer.decode(outputs[0], skip_special_tokens=True)) 

def generate_dataset(variant,model_path,tokenizer_path,ref_string,count):
    start_seq=ref_string
    gen_seq_list=[]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,load_best_model_at_end=True)

    print("\n\n TOKENIZER IMPORTED !!")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("\n\n MODEL IMPORTED !!")
    
    counter=0
    for i in range(0,100): #GENERATING 100 SEQS
        for i in range(0,count):
            text=preprocess_sequence(ref_string)
            gen_sequence=generate_sequence(text,tokenizer,model)
            #gen_seq_list.append([gen_sequence])
            ref_string=gen_sequence
    
        #print("\n\n Sequence Generated !!")    
        gen_seq_list.append([ref_string]) #Appending generated Sequence After Looping 9 times into a List
        counter=counter+1    
        ref_string=start_seq ## Changing back to Reference Genome
        #print("\n\n Sequence Generated !! \n\n Current :",counter) 
    
    #GENERATING A CSV FILE FOR LATER CLASSIFICATION
    df = pd.DataFrame(gen_seq_list,columns =['Seq1'])
    print(df.head())
    
    
    for index, row in df.iterrows():
        df.loc[index, 'Label_Variant'] = variant ## Adding a label column
    
    
    filename="output_"+variant+".csv"
    df.to_csv(filename) 
    
    # with open('output_alpha.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(gen_seq_list)    





generate_dataset("alpha",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_alpha_final/checkpoint-7500",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_alpha_final/checkpoint-7500",
                ref_string,
                9)
print("\n\n Alpha variant Generated")
                
generate_dataset("delta",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_delta_final/checkpoint-27500",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_delta_final/checkpoint-27500",
                ref_string,
                9)
print("\n\n Delta variant Generated")                 
generate_dataset("omicron",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_omicron_final/checkpoint-28000",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_omicron_final/checkpoint-28000",
                ref_string,
                9)
print("\n\n Omicron variant Generated")
generate_dataset("nonvoc",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_nonvoc_final/checkpoint-12000",
                "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/translation_nsp_nonvoc_final/checkpoint-12000",
                ref_string,
                9)
                
print("\n\n NON-VOC variant Generated")



#inputs = tokenizer(text, return_tensors="pt").input_ids


# source_lang = "Seq1"
# target_lang = "Seq2"
# prefix = "translate Seq1 to Seq2: "

# text=prefix+ref_string

# print(text)

# inputs = tokenizer(text, return_tensors="pt").input_ids
# print("\n\n TOKENIZED INPUT !!")







