from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_classification/rbd_classifier_4_variants_final/checkpoint-152500",load_best_model_at_end=True)

print("\n\n TOKENIZER IMPORTED !!")

model = AutoModelForSeq2SeqLM.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_classification/rbd_classifier_4_variants_final/checkpoint-152500")
print("\n\n MODEL IMPORTED !!")

def preprocess_sequence(text_input):
    prefix = "Sequence Variant:"
    return prefix+text_input


df_alpha=pd.read_csv("output_alpha.csv")

print(df_alpha.head())

print(df_alpha.shape)
df_delta=pd.read_csv("output_delta.csv")

print(df_delta.head())

print(df_delta.shape)
df_omicron=pd.read_csv("output_omicron.csv")

print(df_omicron.head())
print(df_omicron.shape)
df_nonvoc=pd.read_csv("output_nonvoc.csv")

print(df_nonvoc.head())
print(df_nonvoc.shape)
# merged_df = pd.merge(df_alpha, df_delta, on="Seq1")
# merged_df = pd.merge(merged_df, df_omicron, on="Seq1")
# merged_df = pd.merge(merged_df, df_nonvoc, on="Seq1")
#df = df.apply(lambda x: np.square(x) if x.name == 'b' else x,axis = 1)

# print(merged_df.head())
# print(merged_df.shape)

merged_df = pd.concat([df_alpha, df_delta, df_omicron, df_nonvoc], axis=0,ignore_index=True)
print(merged_df.head())
print(merged_df.shape)

# merged_df.to_csv("merged.csv")

# from datasets import Dataset

# hf = Dataset.from_pandas(merged_df)



# hf = hf.train_test_split(train_size=0.8, seed=42)

# print("\n Huggingface Dataset : \n",hf)

# hf_clean = hf["train"].train_test_split(train_size=0.8, seed=42)

# print(hf_clean)
# # Rename the default "test" split to "validation"
# hf_clean["validation"] = hf_clean.pop("test")
# # Add the "test" set to our `DatasetDict`

# print(hf_clean)

# hf_clean["test"] = hf["test"]
# print(hf_clean)

# hf=hf_clean

# print(hf)
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
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


for index, row in merged_df.iterrows():
    text=row['Seq1']
    #row['Label_variant']="Alpha"
    #print(row)
    preprocessed_input=preprocess_sequence(text)
    inputs = tokenizer(preprocessed_input, return_tensors="pt")
    #target_label=tokenizer(row['Label_Variant'], return_tensors="pt")
    #outputs = model.generate(inputs)
    tokens = model.generate(**inputs,max_new_tokens=10)
    #print("OUTPUT LABEL:",tokenizer.batch_decode(tokens,skip_special_tokens=True))
    
    #predicted_class_id = logits.argmax().item()
    #print(model.config.id2label[predicted_class_id])
    #print("\n\n Predicted Label : ",tokenizer.decode(outputs[0], skip_special_tokens=True))
    #row["Pred_Variant"]=tokenizer.batch_decode(tokens,skip_special_tokens=True) 
    
    #df.loc[index, 'Label_Variant'] = 'Alpha'
    merged_df.loc[index,'Pred_Variant'] = tokenizer.batch_decode(tokens,skip_special_tokens=True)
    #.loc[index,'Target_label'] = tokenizer.batch_decode(target_label,skip_special_tokens=True)
    #df.set_value(i,'Label_Variant',tokenizer.batch_decode(tokens,skip_special_tokens=True))
    #df.set_value(i,'Pred_Variant',tokenizer.batch_decode(tokens,skip_special_tokens=True))

print(merged_df.head())


merged_df.to_csv("output_all_classified.csv")


