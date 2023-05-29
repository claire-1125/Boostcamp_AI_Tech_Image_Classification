# Boostcamp AI Tech3 Image Clasification Contest (Level 1)

## Contents
- [프로젝트 구조](#프로젝트-구조)
- [Getting Started](#Getting-Started)
   - [Dependencies](#Dependencies)
   - [Install Requirements](#Install-Requirements)
   - [Run](#Run)
   - [Report](#Report)
   - [Acknowledgements](#Acknowledgements)

# 프로젝트 구조

```python
ijkimmy/
├── model/                  # a default directory for saving model output
│  └── loss.py              # loss function classes (e.g focal loss, label smoothing loss, f1 loss)
│  └── model.py             # model class that inherits nn.Module (e.g. PretrainedModels, VGGFace)
│
├── output/                 # a default directory for inference result files 
│
├── project_reports/        # include a review and a report about project timeline and 
│
├── utils/                  # small utility functions
│  └── util.py	            # has functions like EarlyStopping etc.
│
├── data_viz.ipynb          # evaluate model using confusion matrix
│
├── dataset.py              # dataset class that inherits torch.utils.data.Dataset
│
├── ensemble.ipynb          # inference from trained model using hard & soft voting (ensemble)
│
├── inference.py            # inference from trained model (make inference result to csv form)
│
├── run.sh                  # script to run the project 
│
└── train.py                # setting and implementation of training
```


# Getting Started    
## Dependencies
- torch==1.6.0
- torchvision==0.7.0                                                              

## Install Requirements
- `pip install -r requirements.txt`

## Run
- Run the program by modifying the `run.sh` script. By default, it trains three separate ResNet18 models to classify age, gender, and mask using `train.py`, and then creates an output file using `inference.py`.

## Report
- 전반적인 프로젝트 타임라인과 시험내용을 요약해놨습니다. [프로젝트 타임라인](./project_reports/project_timeline.md)
- 대회가 끝난 후 개인 회고를 정리해 놨습니다. [개인회고](./project_reports/afterthoughts.md)

## Acknowledgements
This project is generated from the template [Pytorch-Template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)
