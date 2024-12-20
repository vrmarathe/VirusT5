import pandas as pd
import numpy as np
import nltk
import os
import torch
from transformers import BertTokenizer, TFBertForSequenceClassification, RobertaTokenizer, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from datasets import Dataset
import matplotlib.pyplot as plt
import evaluate
from sklearn import preprocessing
from sklearn import metrics
import argparse

class SequenceClassifier:
    def __init__(self, dataset_path, tokenizer_path, model_checkpoint, training_args):
        self.dataset_path = dataset_path
        self.tokenizer_path = tokenizer_path
        self.model_checkpoint = model_checkpoint
        self.training_args = training_args
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.df = None
        self.hf = None
        self.tokenized_hf = None
        self.trainer = None

    def check_cuda(self):
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

    def load_dataset(self):
        self.df = pd.read_csv(self.dataset_path)
        print("Original Dataset !! ")
        print(self.df.head())
        self.df = self.df.dropna()
        print("\nColumns:", self.df.columns)

    def visualize_variants(self):
        variants = self.df["Variant"].unique()
        variant_counts = [self.df[self.df["Variant"] == variant].shape[0] for variant in variants]
        plt.bar(variants, variant_counts)
        plt.xlabel("Variant")
        plt.ylabel("Count")
        plt.savefig('Variant_count.png')

    def filter_and_prepare_df(self, filtered_labels):
        filtered_df = self.df[self.df['Variant'].isin(filtered_labels)]
        filtered_df['classification'] = filtered_df.apply(lambda row: {'Variant': row['Variant'], 'Sequence': row['Seq1']}, axis=1)
        filtered_df = filtered_df.drop(['Seq1', 'Seq2', 'Variant', 'Lineage', 'GenS1', 'GenS2'], axis=1)
        self.df = filtered_df

    def split_dataset(self):
        self.hf = Dataset.from_pandas(self.df)
        self.hf = self.hf.train_test_split(train_size=0.8, seed=42)
        hf_clean = self.hf["train"].train_test_split(train_size=0.8, seed=42)
        hf_clean["validation"] = hf_clean.pop("test")
        hf_clean["test"] = self.hf["test"]
        self.hf = hf_clean

    def preprocess_function(self, examples, prefix):
        inputs = [prefix + example["Sequence"] for example in examples["classification"]]
        targets = [example["Variant"] for example in examples["classification"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=667, truncation=False)
        return model_inputs

    def tokenize_dataset(self, prefix):
        self.tokenized_hf = self.hf.map(lambda examples: self.preprocess_function(examples, prefix), batched=True)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        le = preprocessing.LabelEncoder()
        le.fit(decoded_preds + decoded_labels)
        numeric_preds = le.transform(decoded_preds)
        numeric_labels = le.transform(decoded_labels)
        result = evaluate.load("accuracy").compute(predictions=numeric_preds, references=numeric_labels)
        return result

    def train_model(self):
        model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint, from_flax=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_checkpoint, max_length=667)
        self.trainer = Seq2SeqTrainer(
            model=model,
            args=self.training_args,
            train_dataset=self.tokenized_hf["train"],
            eval_dataset=self.tokenized_hf["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: self.compute_metrics(eval_preds),
        )
        self.trainer.train()

    def save_training_history(self):
        train_metrics = self.trainer.state.log_history
        train_df = pd.DataFrame(train_metrics)
        train_df.to_csv("metrics_4_varaints_final_8.csv", index=False)

    def evaluate_model(self):
        results = self.trainer.evaluate(eval_dataset=self.tokenized_hf["test"])
        print("\n\n RESULTS OF TEST DATASET !!")
        print("\n\n")
        print(results)

    def run(self):
        self.check_cuda()
        self.load_dataset()
        self.visualize_variants()
        filtered_labels = ['alpha', 'delta', 'omicron', 'nonVOC']
        self.filter_and_prepare_df(filtered_labels)
        self.split_dataset()
        self.tokenize_dataset("Sequence Variant:")
        self.train_model()
        self.evaluate_model()
        self.save_training_history()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RBD Classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, default="rbd_classifier_4_variants_final", help="Output directory")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_total_limit", type=int, default=8, help="Save total limit")
    parser.add_argument("--num_train_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--predict_with_generate", type=bool, default=True, help="Predict with generate")
    parser.add_argument("--fp16", type=bool, default=True, help="Use fp16")

    args = parser.parse_args()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=args.predict_with_generate,
        fp16=args.fp16,
    )
    classifier = SequenceClassifier(
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        model_checkpoint=args.model_path,
        training_args=training_args
    )
    classifier.run()
