import pandas as pd
import numpy as np
import nltk
import warnings
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import evaluate

class TranslationModel:
    def __init__(self, data_path, tokenizer_path, model_checkpoint):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.metric = evaluate.load("bleu")
        self.prefix = "translate Seq1 to Seq2: "
        self.source_lang = "Seq1"
        self.target_lang = "Seq2"
        self.max_length = 667
        self._load_data()
        self._preprocess_data()
        self._tokenize_data()
        self._initialize_model()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path)
        print("Original Dataset !! ")
        print(self.df.head())
        self.df = self.df.dropna()
        print("\nColumns:", self.df.columns)

    def _preprocess_data(self):
        self.df.drop(['Lineage', 'GenS1', 'GenS2'], axis=1, inplace=True)
        self.df['translation'] = self.df.apply(lambda row: {'Seq1': row['Seq1'], 'Seq2': row['Seq2']}, axis=1)
        print(self.df['translation'][0])
        self.df = self.df.drop(['Seq1', 'Seq2', 'Variant'], axis=1)
        self.hf = Dataset.from_pandas(self.df)
        self.hf = self.hf.train_test_split(train_size=0.8, seed=42)
        hf_clean = self.hf["train"].train_test_split(train_size=0.8, seed=42)
        hf_clean["validation"] = hf_clean.pop("test")
        hf_clean["test"] = self.hf["test"]
        self.hf = hf_clean
        print("\n\nDataset after Breaking into Train_Val_Test")
        print(self.hf)

    def _tokenize_data(self):
        def preprocess_function(examples):
            inputs = [self.prefix + example[self.source_lang] for example in examples["translation"]]
            targets = [example[self.target_lang] for example in examples["translation"]]
            model_inputs = self.tokenizer(inputs, text_target=targets, max_length=self.max_length, truncation=False)
            return model_inputs

        self.tokenized_hf = self.hf.map(preprocess_function, batched=True)

    def _initialize_model(self):
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model_checkpoint, max_length=self.max_length)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_checkpoint, from_flax=True)

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
        df_predictions.to_csv("df_predictions_final.csv")
        df_predictions['Labels'] = df_predictions['Labels'].apply(self.seq2kmer, args=(3,))
        df_predictions['Predictions'] = df_predictions['Predictions'].apply(self.seq2kmer, args=(3,))
        decoded_preds = df_predictions['Predictions'].tolist()
        decoded_labels = df_predictions['Labels'].tolist()
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    def train(self):
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
        self.trainer = trainer

    def evaluate(self):
        results = self.trainer.evaluate(eval_dataset=self.tokenized_hf["test"])
        print("\n\n RESULTS OF TEST DATASET !!\n\n")
        print(results)

    def save_training_history(self):
        train_metrics = self.trainer.state.log_history
        train_df = pd.DataFrame(train_metrics)
        train_df.to_csv("metrics_translation_final_4_epochs.csv", index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    data_path = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/RBD_valid_finetuning_nucleotides.csv"
    tokenizer_path = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/virusTrained100k"
    model_checkpoint = "/mmfs1/projects/changhui.yan/vishwajeet.marathe/fine_tune/translation_nsp/virusTrained100k"

    translation_model = TranslationModel(data_path, tokenizer_path, model_checkpoint)
    translation_model.train()
    translation_model.evaluate()
    translation_model.save_training_history()
