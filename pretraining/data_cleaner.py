import datasets
import pandas as pd
from t5_tokenizer_model import SentencePieceUnigramTokenizer
from datasets import load_dataset
from datasets import load_dataset, load_metric
from datasets import Dataset


df_fasta=pd.read_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/processed/dataset200k/train.csv")

print(df_fasta.head())

empty_rows_count = df_fasta.isna().all(axis=1).sum()
print("\nEmpty Rows:",empty_rows_count)

df_fasta=df_fasta.dropna()

df_fasta.to_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/final/dataset200k/train.csv")

print("Cleaning the train dataset done")

df_fasta=pd.read_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/processed/dataset200k/validation.csv")

df_fasta=df_fasta.dropna()

df_fasta.to_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/final/dataset200k/validation.csv")

print("Cleaning the val dataset done")

df_fasta=pd.read_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/processed/dataset200k/test.csv")

df_fasta=df_fasta.dropna()

df_fasta.to_csv("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/final/dataset200k/test.csv")

print("Cleaning the test dataset done")

# print(df_fasta.isna().all(axis=1).sum())

# print("\nEmpty Rows:",empty_rows_count)