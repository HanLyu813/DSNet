# DSNet
Han Lyu, Meijun Sun, Haowei Ran, Yipu Liu, Xinyu Yan, Zheng Wang<br />

The experimental results and prediction maps are available now. The code is currently being organized and will be publicly available later. 

If you are interested in our work, please do not hesitate to contact us at han.lyu.cs@gmail.com via email.

---
> **Abstract:** *Camouflaged Object Detection (COD) has historically been a significant challenge in the field of computer vision. Most existing methods for COD predominantly rely on complex designs to maximize the confidence of foreground regions within spatial features. In contrast, an alternative perspective is that suppressing background distractions to highlight the foreground might be a more effective approach. To address this issue, we propose Distraction Suppression Network, named DSNet. Specifically, Distracion Suppression Module (DSM) is implemented prior to the decoding stage to suppress the distracting information based on the Object-Related Information (ORI) extracted from Object Mining Module (OMM). Then, the Feature Modulation Decoder (FMD) modulates features with varing frequencies and obtains the prediction in a coarse-to-fine way. Experimental results show that our model outperforms existing state-of-the-art models on benchmark datasets by a large margin. Notably, our model maintains its performance even in more complex camouflage scenes.*
>![DSNet3_01](https://github.com/user-attachments/assets/67c16bd8-8569-47a5-9247-b31cc07da2bd)


---

## Usage

### 1. Dataset Preparation
Please organize the dataset folder into the following structure:
```
dataset/
├── TrainDataset/
│   ├── Img/
│   └── GT/
└── TestDataset/
    ├── COD10K/
    ├── NC4K/
    ├── CHAMELEON/
    └── CAMO/
```
- `TrainDataset/Img/` and `TrainDataset/GT/`：Training Set of COD10K
- There are four test set folders under `TestDataset/`, each folder contains the data of the corresponding test set
- You can download the dataset from [[baidu](https://pan.baidu.com/s/1r_mTRJXSeXLCBXpuSgigow),PIN:5j4e]. Please check.
### 2. Training Configuration
- The pretrained PVTv2 is stored in [[baidu](https://pan.baidu.com/s/1bZDnb6bEnAYxBkWLZgU38w),PIN:enh4]. Please place it in the `pretrained/` folder before training.
```bash
python train.py --data_path=dataset/TrainDataset/ --val_path=dataset/TestDataset/ --save_path_train=YOUR_SAVE_PATH --batch_size=36 --recurrence=3 --epochs=50 --lr=0.0001
```
### 3. Testing
```bash
python infer.py --data_path=dataset/TestDataset/ --save_path=YOUR_SAVE_PATH --model_path=YOUR_MODEL_PATH
```
### 4. Evaluation

- Matlab code: One-key evaluation is written in [MATLAB code](https://github.com/DengPingFan/CODToolbox), please follow this the instructions in `main.m` and just run it to generate the evaluation results.

### 4. Results download
![image](https://github.com/user-attachments/assets/38e90928-00c9-4828-8d66-41e40f1ce7ee)


- We provide the prediction maps of our DSNet on four benchmarks: CHAMELEON, CAMO-Test, COD10K-Test, NC4K.<br />
The prediction results of our DSNet are stored in [[baidu](https://pan.baidu.com/s/1mNOlCpobjF-rKoQERMLZtg),PIN:qm3h]. Please check.

## Experimental Results
<img width="1078" alt="image" src="https://github.com/user-attachments/assets/79fa8466-5fa7-4069-bb01-4d80b8439689" />




