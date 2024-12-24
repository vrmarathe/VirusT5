# VirusT5: Harnessing Large Language Models to Predict SARS-CoV-2 Evolution  


## Overview  
VirusT5 is a transformer-based language model built on the T5 architecture, designed to predict SARS-CoV-2 evolution. By modeling viral mutations as a "mutation-as-translation" process, VirusT5 captures mutation patterns in the Receptor-Binding Domain (RBD) of the spike protein, identifies mutation hotspots, and forecasts future viral strains.  

## Features  
- **Variant Classification**: Accurately classifies SARS-CoV-2 variants based on RBD sequences.  
- **Mutation Prediction**: Translates parental RBD sequences into evolved child sequences.  
- **Generative Evolution**: Simulates multi-generational viral evolution.  

## Model Availability
The model is available to use through Huggingface [VirusT5](https://huggingface.co/vrmarathe/VirusT5)
## How To Use The Pretrained Model
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer for the VirusT5 model
tokenizer = AutoTokenizer.from_pretrained("vrmarathe/VirusT5", trust_remote_code=True)

# Load the pre-trained VirusT5 model (T5-based)
model = AutoModelForSeq2SeqLM.from_pretrained("vrmarathe/VirusT5", trust_remote_code=True,from_flax=True)
```



## How It Works  
VirusT5 is pretrained on 100,000 SARS-CoV-2 genome sequences from the GISAID database. Fine-tuning involves tasks like:  
1. Classifying RBD variant types.  
2. Translating parent-child mutation pairs to predict evolutionary changes.  
3. Simulating mutations across multiple viral generations.  

## Performance Highlights  
- **Variant Classification Accuracy**: 97.29%  
- **Mutation Translation BLEU Score**: 0.999  
- **Multi-Generational Evolution Simulation Accuracy**: 100%  

## Installation  
Clone the repository and set up the required dependencies:  
```bash  
git clone https://github.com/vrmarathe/VirusT5.git
cd VirusT5
cd environment
conda env create -f flax2_environment.yml
```
## Datasets  
VirusT5 was trained and fine-tuned using the following datasets:  

### 1. Genome Dataset  
- **Description**: This dataset comprises 100,000 complete SARS-CoV-2 genome sequences, randomly sampled from the GISAID database.  
- **Usage**: Used during the pretraining phase to help the model learn mutation patterns in the SARS-CoV-2 genome.  
- **Details**:  
  - Segmented into non-overlapping sequences of up to 512 base pairs.  
  - Processed using a masked language modeling objective.  
- **Source**: [GISAID Database](https://www.gisaid.org/)
- **Preprocessing Link and Code - https://github.com/deevvan/SARS-CoV-2-transformer-based-model-training-dataset/tree/main

### 2. Receptor Binding Domain (RBD) Dataset  
- **Description**: Contains genetic sequences encoding the receptor-binding domain of the SARS-CoV-2 spike protein.  
- **Usage**:  
  - Fine-tuning for variant classification tasks.  
  - Generating the Parent-Child dataset for evolutionary studies.
  - **Preprocessing For Pretaining and FineTuning Datasets - https://github.com/deevvan/SARS-CoV-2-transformer-based-model-training-dataset/tree/main  
- **Details**:  
  - Codon-aware multiple sequence alignment (MSA) performed using MUSCLE.  
  - Mapped to reference genome (NCBI: NC_004718.3).  

### 3. Parent-Child Dataset  
- **Description**: Contains pairs of RBD sequences where one sequence acts as the evolutionary parent of the other.  
- **Usage**: Fine-tuning for "mutation-as-translation" tasks, where the model predicts the child sequence from the parent sequence.
- - **Preprocessing For Pretaining and FineTuning Datasets - https://github.com/deevvan/SARS-CoV-2-transformer-based-model-training-dataset/tree/main  
- **Details**:  
  - Constructed from RBD sequences divided into 10 temporal bins.  
  - Includes 500,000 parent-child pairs sampled across Alpha, Delta, Omicron, and non-VOC variants.
    
    

### Notes  
- **Access**: While the datasets rely on public resources like GISAID, access may require registration or compliance with their terms of use.  
- **Preprocessing**: Preprocessing scripts for dataset preparation are available in the Preprocessing in [Pretaining and FineTuning Datasets directory](https://github.com/deevvan/SARS-CoV-2-transformer-based-model-training-dataset/tree/main).
- Datasets will be provided on request.
## Pretraining and Fine-Tuning  

### Pretraining  
VirusT5 was pretrained on a large corpus of SARS-CoV-2 genome sequences to learn the underlying syntax and grammar of genomic data.  
- **Dataset**: Genome Dataset comprising 100,000 SARS-CoV-2 genome sequences from GISAID.  
- **Objective**: Masked Language Modeling (MLM) with 15% token masking using sentinel tokens.  
- **Sequence Length**: Segmented into sequences of up to 512 base pairs.  
- **Optimization**:  
  - Inverse square root learning rate schedule.  
  - Initial learning rate: 0.005 for 2,000 steps, followed by exponential decay.  
- **Training Hardware**:  
  - NDSU CCAST HPC clusters with 32 CPU cores, 100 GB RAM, and two NVIDIA A40 GPUs (40 GB each).  
- **Duration**: Pretrained for 12,000 steps.
- The scripts for the pretraining can be found in the pretraining folder  

### Fine-Tuning  
Fine-tuning tailored the pretrained VirusT5 model for specific downstream tasks, such as classification and mutation prediction.  
#### Tasks  
1. **Variant Classification**:  
   - **Dataset**: RBD Dataset, divided into training (60%), validation (20%), and test (20%) sets.  
   - **Objective**: Predict variant types (e.g., Alpha, Delta, Omicron, non-VOC) from RBD sequences.  
   - **Result**: Achieved 97.29% accuracy.
   - The original finetuning script for RBD classification can be found in the rbd-classification folder [rbd-classifier](https://github.com/vrmarathe/VirusT5/tree/1d290a99f767fb5cb4bfd598b5fff7e1b348138a/rbd-classifier).
   - The general classifier  script can be used for other classification experiments can be found in [General Classification](https://github.com/vrmarathe/VirusT5/blob/1d290a99f767fb5cb4bfd598b5fff7e1b348138a/rbd-classifier/classifier-general.py)
      

2. **Mutation Translation**:  
   - **Dataset**: Parent-Child Dataset with 500,000 RBD sequence pairs representing evolutionary parent-child relationships.  
   - **Objective**: Predict how an RBD sequence evolves from one generation to the next.
   - The original finetuning script for RBD translation/evolution predication can be found in the [RBD-translation](https://github.com/vrmarathe/VirusT5/tree/1d290a99f767fb5cb4bfd598b5fff7e1b348138a/rbd-translation).
   - The general mutation translation  script can be used for other experiments and can be found in [Translation-general](https://github.com/vrmarathe/VirusT5/blob/1d290a99f767fb5cb4bfd598b5fff7e1b348138a/rbd-translation/translation-general.py)
   - **Evaluation**:  
     - BLEU Score: 0.999  
     - Sequence Identity: 99.97% ± 0.1%
3. **For Other Tasks**
     - The model is based on the T5 archictecture. The model can be fine-tuned to similar DNA/Genome/Virus related tasks that T5 was fine-tned on like summarization,question-answering etc. 

#### Fine-Tuning Process  
- The model was trained and validated over multiple epochs until convergence, stopping when both training and validation losses stabilized.  
- The following split was used for all datasets:  
  - **Training**: 60%  
  - **Validation**: 20%  
  - **Testing**: 20%  
- Fine-tuning used similar hardware as pretraining.

  
## Citation  
If you use VirusT5 in your research, please cite the following paper:
```
@misc{marathe2024virust5harnessinglargelanguage,
      title={VirusT5: Harnessing Large Language Models to Predicting SARS-CoV-2 Evolution}, 
      author={Vishwajeet Marathe and Deewan Bajracharya and Changhui Yan},
      year={2024},
      eprint={2412.16262},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2412.16262}, 
}
```






