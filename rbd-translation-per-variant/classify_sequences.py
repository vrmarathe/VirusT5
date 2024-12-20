from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import pandas as pd


tokenizer = AutoTokenizer.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/rbd_classifier/checkpoint-163000",load_best_model_at_end=True)

print("\n\n TOKENIZER IMPORTED !!")

model = AutoModelForSeq2SeqLM.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/rbd_classifier/checkpoint-163000")
print("\n\n MODEL IMPORTED !!")

def preprocess_sequence(text_input):
    prefix = "Sequence Variant:"
    return prefix+text_input


df=pd.read_csv("output_Alpha.csv")

print(df.head())

#df = df.apply(lambda x: np.square(x) if x.name == 'b' else x,axis = 1)




for index, row in df.iterrows():
    text=row['Seq1']
    #row['Label_variant']="Alpha"
    print(row)
    preprocessed_input=preprocess_sequence(text)
    inputs = tokenizer(preprocessed_input, return_tensors="pt")
    #outputs = model.generate(inputs)
    tokens = model.generate(**inputs,max_new_tokens=10)
    #print("OUTPUT LABEL:",tokenizer.batch_decode(tokens,skip_special_tokens=True))
    
    #predicted_class_id = logits.argmax().item()
    #print(model.config.id2label[predicted_class_id])
    #print("\n\n Predicted Label : ",tokenizer.decode(outputs[0], skip_special_tokens=True))
    #row["Pred_Variant"]=tokenizer.batch_decode(tokens,skip_special_tokens=True) 
    
    #df.loc[index, 'Label_Variant'] = 'Alpha'
    df.loc[index, 'Pred_Variant'] = tokenizer.batch_decode(tokens,skip_special_tokens=True)
    #df.set_value(i,'Label_Variant',tokenizer.batch_decode(tokens,skip_special_tokens=True))
    #df.set_value(i,'Pred_Variant',tokenizer.batch_decode(tokens,skip_special_tokens=True))

print(df.head())
df.to_csv("output_alpha_classified.csv")


