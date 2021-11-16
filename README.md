# ADAPT_Speech

## Setup
- Install libraries from requirements.txt
- Datasets can be downloaded from the sources mentioned in the paper. Language dataset mapping (Italian:EMOVO,Persian:ShEMO, Urdu: Urdu SER)
- Install torchaudio:

			pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
 - For Hyperbolic implementation we use implementations from Hazy Research: https://github.com/HazyResearch/hgcn. Clone the hgcn repository and enter inside the directory

      git clone https://github.com/HazyResearch/hgcn.git 
      
      cd hgcn

## Training



