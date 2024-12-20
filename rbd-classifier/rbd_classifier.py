#Importing Libraries

import pandas as pd
import numpy as np
import nltk
import os
import torch

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

for i in range(num_gpus):
    device = torch.device(f"cuda:{i}")
    print(f"GPU {i}:")
    print(f"  Name: {torch.cuda.get_device_name(i)}")
    print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")


#import tensorflow as tf
#sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer
import warnings

import matplotlib.pyplot as plt

#Importing datasets

df = pd.read_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/RBD_valid_finetuning_nucleotides.csv")
print("Original Dataset !! ")
print(df.head())

df=df.dropna()



print("\nColumns:",df.columns)

#Visulization

variants = df["Variant"].unique()

# Create a list of counts for each variant
variant_counts = []
for variant in variants:
    count = df[df["Variant"] == variant].shape[0]
    variant_counts.append(count)

# Create a bar plot of the variant counts
plt.bar(variants, variant_counts)
plt.xlabel("Variant")
plt.ylabel("Count")

plt.savefig('Variant_count.png')


# Filter for specific labels
filtered_labels = ['alpha', 'delta', 'omicron', 'nonVOC']
filtered_df = df[df['Variant'].isin(filtered_labels)]

df=filtered_df




df['classification'] = df.apply(lambda row: {'Variant': row['Variant'], 'Sequence': row['Seq1']}, axis=1)

print(df.head())
#print(df['classification'][0])




df = df.drop(['Seq1', 'Seq2','Variant'], axis=1)

df.drop(['Lineage','GenS1','GenS2'], axis=1,inplace=True)

print(df.head())

#Breaking the dataset into train-val-test
from datasets import Dataset

hf = Dataset.from_pandas(df)

hf = hf.train_test_split(train_size=0.8, seed=42)

print("\n Huggingface Dataset : \n",hf)

hf_clean = hf["train"].train_test_split(train_size=0.8, seed=42)

print(hf_clean)
# Rename the default "test" split to "validation"
hf_clean["validation"] = hf_clean.pop("test")
# Add the "test" set to our `DatasetDict`

print(hf_clean)

hf_clean["test"] = hf["test"]
print(hf_clean)

hf=hf_clean

print("\n\nDataset after Breaking into Train_Val_Test")
print(hf)


#Preprocessing the dataset

print("\n\n Tokenization")
tokenizer = tokenizer = AutoTokenizer.from_pretrained('/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_classification/virusTrained100k')

source_lang = "Sequence"
target_lang = "Variant"
prefix = "Sequence Variant:"


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["classification"]]
    targets = [example[target_lang] for example in examples["classification"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=667, truncation=False)
    return model_inputs

tokenized_hf = hf.map(preprocess_function, batched=True)

#Tokenization and training

checkpoint="/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/virusTrained100k"

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint,max_length=667)

# Initialize the model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,FlaxT5Model,T5ForConditionalGeneration
#num_labels = len(df['Variant'].unique())
model = T5ForConditionalGeneration.from_pretrained(checkpoint, from_flax=True)
#model = TFBertForSequenceClassification.from_pretrained('/content/drive/MyDrive/Research_Summer_2023/virusTrained100k', num_labels=num_labels)


from transformers import AdamWeightDecay

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,from_flax=True)



print("\n\nIMPORTED MODEL !! \n ")


import evaluate
# Setup evaluation
nltk.download("punkt", quiet=True)
#metric = evaluate.load("rouge")
metric = evaluate.load("accuracy")
import numpy as np


import numpy as np
from sklearn import metrics



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

from sklearn import preprocessing

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    #print("\n\n I AM HERE !! ")
    if isinstance(preds, tuple):
        preds = preds[0]
    #logits, labels = eval_preds
    #predictions = np.argmax(logits, axis=-1)
    #return accuracy.compute(predictions=predictions, references=labels)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print("\n Decoded Preds: ",decoded_preds,"\n")
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print("\n Decoded Labels: ",decoded_labels,"\n")
    
    le = preprocessing.LabelEncoder()
    #le.fit(decoded_labels)
    le.fit(decoded_preds + decoded_labels)
    #le.fit(decoded_preds)
    print("\n Classes:",list(le.classes_))
    
    numeric_preds=le.transform(decoded_preds)
    numeric_labels=le.transform(decoded_labels)
    
    print(le.inverse_transform(numeric_preds[0:5]))
    print(le.inverse_transform(numeric_labels[0:5]))
    
    print("\n Numeric Preds:",numeric_preds)
    print("\n Numeric Labels:",numeric_labels)
    
     # rougeLSum expects newline after each sentence
    #decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    #decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    
    
    #result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = metric.compute(predictions=numeric_preds, references=numeric_labels)
    print(result)
    return result
    
    
print("Training STARTED !!")



training_args = Seq2SeqTrainingArguments(
    output_dir="rbd_classifier_4_variants_final",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=8,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=True,
    #push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_hf["train"],
    eval_dataset=tokenized_hf["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()




results = trainer.evaluate(eval_dataset=tokenized_hf["test"])

print("\n\n Extracting Training History !! \n ")
train_metrics = trainer.state.log_history
#eval_metrics = trainer.state.eval_history

# Convert to Pandas DataFrames
train_df = pd.DataFrame(train_metrics)
#eval_df = pd.DataFrame(eval_metrics)

# Save to CSV
train_df.to_csv("metrics_4_varaints_final_8.csv", index=False)
#eval_df.to_csv("eval_metrics.csv", index=False)


print("\n\n RESULTS OF TEST DATASET !!")
print("\n\n")

print(results)

#Evaluation


