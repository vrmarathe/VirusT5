import pandas as pd
import numpy as np
import nltk

#import tensorflow as tf
#sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, RobertaTokenizer
from transformers import AutoTokenizer
import warnings

import matplotlib.pyplot as plt
# Disable tqdm warning message
warnings.filterwarnings("ignore", category=UserWarning)

# Load the data
df = pd.read_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/RBD_valid_finetuning_nucleotides.csv")
print("Original Dataset !! ")
print(df.head())

df=df.dropna()



print("\nColumns:",df.columns)

# for column in df.columns:
#     unique_values = df[column].unique()
    
#     plt.hist(unique_values, bins=len(unique_values))
#     plt.title(f'Unique Values Histogram - {column}')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.savefig('Unique_hist.png')

# # Get the unique variants
# lineage = df["Lineage"].unique()

# # # Create a list of counts for each variant
# lineage_counts = []
# for lin in lineage:
#     count = df[df["Lineage"] == lin].shape[0]
#     lineage_counts.append(count)

# # # Create a bar plot of the variant counts
# plt.bar(lineage, lineage_counts)
# plt.xlabel("Lineage")
# plt.ylabel("Count")

# plt.savefig('Lineage_count.png')

##DATA PREPROCESSING


df.drop(['Lineage','GenS1','GenS2'], axis=1,inplace=True)

df['translation'] = df.apply(lambda row: {'Seq1': row['Seq1'], 'Seq2': row['Seq2']}, axis=1)

print(df['translation'][0])

df = df.drop(['Seq1', 'Seq2','Variant'], axis=1)

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



source_lang = "Seq1"
target_lang = "Seq2"
prefix = "translate Seq1 to Seq2: "


tokenizer = tokenizer = AutoTokenizer.from_pretrained('/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/virusTrained100k')


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=667, truncation=False)
    return model_inputs


tokenized_hf = hf.map(preprocess_function, batched=True)



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

metric = evaluate.load("bleu")


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers
    



#metric = evaluate.load("sacrebleu")

import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels




# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     predictions = np.argmax(logits, axis=-1)
    
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result


# https://huggingface.co/docs/evaluate/v0.4.0/en/transformers_integrations#seq2seqtrainer
#counter=1
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    
    df_predictions = pd.DataFrame({'Labels': decoded_labels, 'Predictions':decoded_preds})
    print(df_predictions.head())
    
    #filename="df_predictions_"+str(counter)+".csv"
    df_predictions.to_csv("df_predictions_final.csv")
     
    df_predictions['Labels'] = df_predictions['Labels'].apply(seq2kmer,args=(3,))
    df_predictions['Predictions'] = df_predictions['Predictions'].apply(seq2kmer,args=(3,))
    
    decoded_preds = df_predictions['Predictions'].tolist()
    decoded_labels = df_predictions['Labels'].tolist()
    
    decoded_preds = list(map(seq2kmer(seq, 3), decoded_preds))
    decoded_labels = list(map(seq2kmer(seq, 3), decoded_labels))
    
    
    df_predictions = pd.DataFrame({'Labels': decoded_labels, 'Predictions:':decoded_preds})
    #print(df_predictions.head())
    
    #df_predictions.to_csv("df_predictions.csv")
    
    #df_predictions['Labels'] = df_cleaner['Labels'].apply(seq2kmer,args=(3,))
    #df_predictions['Predictions'] = df_cleaner['Predictions'].apply(seq2kmer,args=(3,))
    
    
    
    #df_predictions.to_csv("df_predictions.csv")

    #print("\n\n Decoded Preds:",decoded_preds[0],"\n\n Decoded Labels:",decoded_labels[0])
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    #counter=counter+1
    return result




# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
    
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     #return accuracy.compute(predictions=predictions, references=labels)
    
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     #print("\n Decoded Preds: ",decoded_preds,"\n")
    
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     #print("\n Decoded Labels: ",decoded_labels,"\n")
    
    
#     # rougeLSum expects newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    
    
#     #print("\nDecoded Predictions:\n ",decoded_preds)
#     #print("\nDecoded Labels:\n ",decoded_labels)
    
    
#     #import pandas as pd
#     df_predictions = pd.DataFrame({'Labels': decoded_labels, 'Predictions:':decoded_preds})
#     print(df_predictions.head())
#     df_predictions.to_csv("df_predictions.csv")
    
    
    
    
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     print(result)
    
#     return result
    




# import evaluate
# # # Setup evaluation
# nltk.download("punkt", quiet=True)
# metric = evaluate.load("rouge")
# # metric = evaluate.load("accuracy")
# import numpy as np


# import numpy as np
# from sklearn import metrics



# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]
#     return preds, labels

# from sklearn import preprocessing

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
    
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
    #return accuracy.compute(predictions=predictions, references=labels)
    
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     print("\n Decoded Preds: ",decoded_preds,"\n")
    
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     print("\n Decoded Labels: ",decoded_labels,"\n")
    
#     le = preprocessing.LabelEncoder()
#     #le.fit(decoded_labels)
#     le.fit(decoded_preds)
#     print("\n Classes:",list(le.classes_))
    
#     numeric_preds=le.transform(decoded_preds)
#     numeric_labels=le.transform(decoded_labels)
    
#     print(le.inverse_transform(numeric_preds[0:5]))
#     print(le.inverse_transform(numeric_labels[0:5]))
    
#     print("\n Numeric Preds:",numeric_preds)
#     print("\n Numeric Labels:",numeric_labels)
    
#      # rougeLSum expects newline after each sentence
#     #decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     #decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    
    
#     #result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     result = metric.compute(predictions=numeric_preds, references=numeric_labels)
#     print(result)
#     return result
    
#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     print("\nPred:",decoded_preds,"\nLabel:",decoded_labels)
#     print(accuracy.compute(predictions=decoded_preds, references=decoded_labels))
#     return accuracy.compute(predictions=decoded_preds, references=decoded_labels)
    
#     print("\n\n Accurary:",metrics.accuracy_score(decoded_labels, decoded_preds))
    
#     result = metrics.accuracy_score(decoded_labels, decoded_preds)
#     print(metrics.classification_report(decoded_labels, decoded_preds))
    
#     return result
    
    
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result
    

    

training_args = Seq2SeqTrainingArguments(
    output_dir="translation_nsp_final",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
    generation_max_length=667,
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

# Evaluation using sklearn Metrics


results = trainer.evaluate(eval_dataset=tokenized_hf["test"])

print("\n\n Extracting Training History !! \n ")
train_metrics = trainer.state.log_history
#eval_metrics = trainer.state.eval_history

# Convert to Pandas DataFrames
train_df = pd.DataFrame(train_metrics)
#eval_df = pd.DataFrame(eval_metrics)

# Save to CSV
train_df.to_csv("metrics_translation_final_4_epochs.csv", index=False)


#eval_df.to_csv("eval_metrics.csv", index=False)

print("\n\n RESULTS OF TEST DATASET !!")
print("\n\n")

print(results)




