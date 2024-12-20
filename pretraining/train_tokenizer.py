import datasets
import pandas as pd
from t5_tokenizer_model import SentencePieceUnigramTokenizer
from datasets import load_dataset
from datasets import load_dataset, load_metric
from datasets import Dataset

vocab_size = 32_000
input_sentence_size = None

# Initialize a dataset
#dataset = datasets.load_dataset("oscar", name="unshuffled_deduplicated_no", split="train")

#from datasets import load_from_disk
#dataset = load_from_disk("/mmfs1/projects/changhui.yan/vishwajeet.marathe/virusTransformer/data/final/datasetFinal")

df_fasta=pd.read_csv("//mmfs1//projects//changhui.yan//vishwajeet.marathe//virusTransformer//data//final//dataset200k//train.csv")
train_dataset = Dataset.from_pandas(df_fasta,split='train')

#train_dataset = train_dataset.rename_column("Segment", "Sequence")
print("Imported Dataset!")
print(train_dataset)
# Sequences = [
#     doc for doc in hf_fasta_final["train"]["Sequence"] if len(doc) > 0
# ]


#train_dataset=dataset['train']
#print
tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

print(train_dataset)

# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(train_dataset)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield train_dataset[i: i + batch_length]["Sequence"]

print(train_dataset)


print("BEFORE TRAINING !!")
# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)

print("I AM HERE ! AFTER TRAINING")


# Save files to disk
tokenizer.save("./virusTokenizer200k/tokenizer.json")