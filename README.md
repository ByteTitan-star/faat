
This is the official implementation of our paper [A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers](https://arxiv.org/abs/2509.19947), accepted by NeurIPS 2025. 

## Requirements
conda create --name GeneralComponents python=3.8

conda activate GeneralComponents

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install numpy==1.22.3

pip install jupyter_core==5.8.1

pip install opencv-python==4.5.5.64

pip install scipy==1.10.1

## A Quick Start
**Step 1: Calculate metrics for sample selection**

```
CUDA_VISIBLE_DEVICES=0 python cal_metric.py --output_dir save_metric
```
We also provide the result in ./resource/save_metric_10_res
**Step 2: Train backdoored model with different sample selection methods**

```
CUDA_VISIBLE_DEVICES=0 python train_backdoor.py --output_dir save_metric --result_dir save_res --selection res --res_sel linear --backdoor_type badnets --y_target 0 --select_epoch 10 --poison_rate 0.01 --type 0:0:0
```
>- res-X in the paper : e.g., res-linear => --selection res --res_sel linear

>- forgetting event, grad, ... : e.g., forget => --selection forget

>- stealth meaning Component B: e.g., Component B => --selection stealth
>>- We provide the application of Component B upon Blended attacks with GMSD. Application to other attacks with other metrics is similar to the provided codes.


type: 
>- for Badnets, 0:0:0, 1:1:1, 2:2:2, 0:1:1 used in paper. e.g. 0:0:0 => --type 0:0:0
      
>- for Blend, 2:1:3, 2:2:2 used in paper. e.g. 2:1:3 => --type 2:1:3
      
>- for Quantize, 24:28:8 used in paper. e.g. 24:28:8 => --num_levels 24:28:8

Currently, Component C has been adapted for Badnets (type), Blend (type), and Quantize (num_levels). For application to other attacks, modifications can be made by referring to the aforementioned attacks. The meaning of "type" varies for each attack and is related to its specific definition.

`--select epoch` specifies the epoch used to calculate the statistics. `--output_dir` must be the same with the one used in cal_metric.py

For the backdoor defense experiments, the implement is based on the Backdoorbench repository (https://github.com/SCLBD/BackdoorBench.git). We simply save the indices filtered out from the train_backdoor.py and replace the indices in the corresponding code in BackdoorBench to test the defense (random selection is used by default).

## Citing
If this work or our codes are useful for your research, please kindly cite our paper as follows.

```
@misc{wu2025setgeneralizedcomponentsachieve,
      title={A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers}, 
      author={Zhixiao Wu and Yao Lu and Jie Wen and Hao Sun and Qi Zhou and Guangming Lu},
      year={2025},
      eprint={2509.19947},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
}
```
