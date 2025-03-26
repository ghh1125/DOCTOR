## File Structure
```shell
.
├── README  # * Instruction to this repo
├── requirements  # * Requirements for Conda Environment
├── data  # * Place data split & preprocessed data
├── models  # * Codes for FakingRecipe Model
├── utils  # * Codes for Training and Inference
├── main  # * Codes for Training and Inference
└── run  # * Codes for Training and Inference  
```

## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
- You can download preprocessed features and checkpoints in [this repo](https://github.com/ICTMCG/FakingRecipe). 

## Environment
```shell
  conda create -n aaa python=3.9
  conda activate aaa
  pip install -r requirements.txt
```
## Data Preprocess
```shell
  # You can classify the data into different domains by this commend
  python domain_classify.py
```
- To facilitate access to domain-specific data, we provide domain-partitioned datasets (fakesv and fakett) in the data folder FakeSV_Domain_output and FakeTT_Domain_output.
## Quick Start
You can train and test by following code:
 ```shell
  # First, select different training and testing domains based on your criteria in domain_split.py, then run it to generate the splits.
  python domain_split.py
  
  # Then you can fast run by this:
  ./run.sh
  
  # Or you can run like this:
  # FakeSV:
  python main.py --dataset fakesv --mode train --inference_ckp ./provided_ckp/FakingRecipe_fakesv --dg --diffusion --alpha 0.1 --beta 3 --gamma 0.05 --lr 5e-5
  # FakeTT:
  python main.py --dataset fakett --mode train --inference_ckp ./provided_ckp/FakingRecipe_fakett --dg --diffusion --alpha 0.1 --beta 3 --gamma 0.05 --lr 1e-3
  ```

