# ADAPT_Speech

## Setup
- Install libraries from requirements.txt
- Datasets can be downloaded from the sources mentioned in the paper. Language dataset mapping (Italian:EMOVO,Persian:ShEMO, Urdu: Urdu SER)
- Install torchaudio:

			pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
 - For Hyperbolic implementation we use implementations from Hazy Research: https://github.com/HazyResearch/hgcn. Clone the hgcn repository and enter inside the directory

      git clone https://github.com/HazyResearch/hgcn.git 
      
      cd hgcn
      
- The datasets can be downloaded from the respective sources mentioned in the paper. Load the datasets in the respective folders mentioned below as this is the format followed by (dataset)_train.csv for reading the speech files.

	- URDU: create a folder named 'urdu-language-speech-dataset', load data here.
	- ShEMO: create a folder named 'shemo-persian-speech-emotion-detection-database', load data here.
	- EMOVO: create a folder named 'emovo', load data here.
	
- For data loading, create a folder named 'data' and paste the dataset specific files taken from 'data_files' folder. Similarly vocab.json can also be replaced with the dataset specific file
Example command:	!cp /data_files/train_shemo.csv /data/train.csv

## Training
- model_run.py includes processing and trainging functions used by the model.
- For trying different hyperbolic variants:(HVIB, HVIB-C and ADAPT-VIB), change self.c in the function Wav2Vec2ClassificationHeadViBERTHyperbolic() {lines:575-582} accordingly. Dataset-specific hyperbolicity can also be changed here.
- To run base model or VIB replace trainer_hyp_vib.train() with trainer_base.train() and trainer_hyp_vib.train() respectively.


