from sklearn import preprocessing
import numpy as np
from sklearn import metrics
import pandas as pd

df=pd.read_csv("output_all_classified_final.csv")

print(df.head())


label_mapping = {
    "aha":'alpha',
    "dta":'delta',
    "mcrn": 'omicron',
    "nnvc": 'nonvoc',
    "gamma": 'gamma',
    "ita": 'iota',
    "kaa": 'kappa',
    "ambda": 'lambda',
    "ta": 'zeta',
    "mu": 'm',
    "bta": 'beta',
    "epsilon": 'epsilon',
    "thta": 'theta'
}
print(df.head())
print(df.loc[df["Label_Variant"]=="nan"])
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
#Classes: ['aha', 'ambda', 'bta', 'dta', 'gamma', 'kaa', 'm', 'mcrn', 'nnvc', 'sn', 'ta', 'thta', 'whanh']
#Classes: ['aha', 'ambda', 'bta', 'dta', 'gamma', 'kaa', 'm', 'mcrn', 'nnvc', 'sn', 'ta', 'thta', 'whanh']
#df['Label_Variant'] = df['Label_Variant'].map(label_mapping)
df['Pred_Variant'] = df['Pred_Variant'].map(label_mapping)
print(df.head())
print(df.shape)
#import evaluate
from sklearn.metrics import classification_report
# Setup evaluation
#nltk.download("punkt", quiet=True)
#metric = evaluate.load("rouge")
#metric = evaluate.load("accuracy")
le = preprocessing.LabelEncoder()
    #le.fit(decoded_labels)
preds=df['Pred_Variant'].tolist()
labels=df['Label_Variant'].tolist()
le.fit(preds+labels)
    #le.fit(decoded_preds)
print("\n Classes:",list(le.classes_))
    
numeric_preds=le.transform(df['Pred_Variant'])
numeric_labels=le.transform(df['Label_Variant'])
    
#print(le.inverse_transform(numeric_preds[0:5]))
#print(le.inverse_transform(numeric_labels[0:5]))
    
#print("\n Numeric Preds:",numeric_preds)
#print("\n Numeric Labels:",numeric_labels)
    
     # rougeLSum expects newline after each sentence
    #decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    #decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    
    
    #result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
# result = metric.compute(predictions=numeric_preds, references=numeric_labels)
# print("\n\n ACCURACY USING HUGGINGFACE EVALUATE:",result)

cm = confusion_matrix(numeric_labels,numeric_preds)

sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=le.classes_,
            yticklabels=le.classes_,cmap='Blues')
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix For Virus Evolution',fontsize=17)
plt.savefig("confusion_matrix.png")

#print(classification_report(numeric_labels, numeric_preds, target_names=numeric_labels))

