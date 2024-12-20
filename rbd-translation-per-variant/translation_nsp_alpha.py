import pandas as pd
import numpy as np
import nltk
import warnings
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
import evaluate

# Disable tqdm warning message
warnings.filterwarnings("ignore", category=UserWarning)

class TranslationModel:
    def __init__(self, data_path, tokenizer_path, model_checkpoint, output_dir, variant):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.model_checkpoint = model_checkpoint
        self.output_dir = output_dir
        self.variant = variant
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.metric = evaluate.load("bleu")
        self.prefix = "translate Seq1 to Seq2: "
        self.source_lang = "Seq1"
        self.target_lang = "Seq2"
        self.max_length = 667

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        df_variant = df.loc[df["Variant"] == self.variant]
        df_variant.drop(['Lineage', 'GenS1', 'GenS2'], axis=1, inplace=True)
        df_variant['translation'] = df_variant.apply(lambda row: {'Seq1': row['Seq1'], 'Seq2': row['Seq2']}, axis=1)
        df_variant = df_variant.drop(['Seq1', 'Seq2', 'Variant'], axis=1)
        hf = Dataset.from_pandas(df_variant)
        hf = hf.train_test_split(train_size=0.8, seed=42)
        hf_clean = hf["train"].train_test_split(train_size=0.8, seed=42)
        hf_clean["validation"] = hf_clean.pop("test")
        hf_clean["test"] = hf["test"]
        self.hf = hf_clean

    def preprocess_function(self, examples):
        inputs = [self.prefix + example[self.source_lang] for example in examples["translation"]]
        targets = [example[self.target_lang] for example in examples["translation"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.max_length, truncation=False)
        return model_inputs

    def tokenize_data(self):
        self.tokenized_hf = self.hf.map(self.preprocess_function, batched=True)

    def initialize_model(self):
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint, from_flax=True)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_checkpoint, max_length=self.max_length)

    def seq2kmer(self, seq, k):
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        df_predictions = pd.DataFrame({'Labels': decoded_labels, 'Predictions': decoded_preds})
        df_predictions.to_csv("df_predictions.csv")
        
        df_predictions['Labels'] = df_predictions['Labels'].apply(self.seq2kmer, args=(3,))
        df_predictions['Predictions'] = df_predictions['Predictions'].apply(self.seq2kmer, args=(3,))
        
        decoded_preds = df_predictions['Predictions'].tolist()
        decoded_labels = df_predictions['Labels'].tolist()

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    def train_model(self):
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=5,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True,
            generation_max_length=self.max_length,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_hf["train"],
            eval_dataset=self.tokenized_hf["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        results = trainer.evaluate(eval_dataset=self.tokenized_hf["test"])
        print("\n\n RESULTS OF TEST DATASET !!\n\n")
        print(results)

        train_metrics = trainer.state.log_history
        train_df = pd.DataFrame(train_metrics)
        train_df.to_csv(f"metrics_translation_{self.variant}_4_epochs.csv", index=False)

if __name__ == "__main__":
    data_path = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/RBD_valid_finetuning_nucleotides.csv"
    tokenizer_path = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/virusTrained100k"
    model_checkpoint = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/rbd_per_variant_nsp/virusTrained100k"
    output_dir = "translation_nsp_variant_final"
    variant = "alpha"  # Change this to the desired variant

    translation_model = TranslationModel(data_path, tokenizer_path, model_checkpoint, output_dir, variant)
    translation_model.load_data()
    translation_model.tokenize_data()
    translation_model.initialize_model()
    translation_model.train_model()
