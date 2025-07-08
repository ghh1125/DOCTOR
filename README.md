# Consistent and Invariant Generalization Learning for Short-video Misinformation Detection


🚀 **Exciting News**! 

✨ We are **thrilled** to announce that our paper, titled **"Consistent and Invariant Generalization Learning for Short-video Misinformation Detection"**, has been **accepted** for presentation at **ACM MM 2025**! 🎉
🎉
---

# Quick Start

```shell

├── README 
├── requirements.txt
├── FakeTT_Domain_output
├── FakeSV_Domain_output 
├── data 
├── models
├── utils
├── provided_ckp
├── fea
├── main.py
├── run.sh
└── run.py
```

## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
- You can download preprocessed features and checkpoints in [this repo](https://github.com/ICTMCG/FakingRecipe). 

## Environment
```shell
  conda create -n DOCTOR python=3.9
  conda activate DOCTOR
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

# Citation

```
@misc{guo2025consistentinvariantgeneralizationlearning,
      title={Consistent and Invariant Generalization Learning for Short-video Misinformation Detection}, 
      author={Hanghui Guo and Weijie Shi and Mengze Li and Juncheng Li and Hao Chen and Yue Cui and Jiajie Xu and Jia Zhu and Jiawei Shen and Zhangze Chen and Sirui Han},
      year={2025},
      eprint={2507.04061},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.04061}, 
}
```
