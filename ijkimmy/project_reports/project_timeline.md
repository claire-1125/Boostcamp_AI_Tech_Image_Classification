## Environment Setup

- **AI Stage 통해 서버 생성 (2/21)**  
초반에는 서버 내의 JupyterLab을 사용해 코드 작업을 하였으나 2/23일 수요일날 baseline코드를 부여 받고 vscode내에서 ssh해서 파이썬 & 노트북 사용하여 코드 작업을 진행하였습니다. Pytorch Template과 같은 구조를 처음 보는 터라 낯설긴 하였으나 익숙해 지고 난 뒤 생각해보니 위와 같은 구조를 사용해서 프로젝트를 만드는 것이 주피터 노트북을 사용한 코드보다 가독성과 코드 재사용면에서 우월하다고 생각합니다.
- **Baseline code & wandb setup (2/23)**  
원활한 실험 공유를 위해 wandb를 팀적으로 세팅하였습니다. 그 과정에서 wandb와 tensorboard를 sync해주었습니다 ([참고](https://docs.wandb.ai/guides/integrations/tensorboard)).

## Data Analysis (EDA)

- **각 클래스별 분포 확인해보기 (2/21)**  
주어진 데이터 내에서 한 개인이 가지고 있는 사진은 마스크 미착용 1장, 오착용 1장, 착용 5장으로 마스크를 착용한 사진의 분포가 월등히 많았습니다. 더불어, 연령대 별 분포 내에서 불균형이 심하게 있었습니다 (30이하 > 30~60 > 60이상). 여성 사진이 남성 사진보다 조금 더 많이 있었으나 마스크나 연령대 분포만큼 불균형이 심하지 않았습니다. 더 나아가 남성의 경우 청년층의 비율이 높고, 여성의 경우 반대로 중·장년층의 비율이 높다는 것을 알게 되었습니다. 
각 클래스별 분포의 불균형을 해결하는 것이 대회의 주 요점이었습니다.
- **이미지 크기 (2/21)**  
주어진 이미지의 크기는 (384, 512)로 세로로 긴 형태의 이미지였습니다. 주어진 이미지의 크기가 크기에 원활한 실험을 위해 전처리 과정에서 이미지를 줄여주는게 더 효과적일 것 같았습니다.
- **분석 대상이 되는 객체의 위치 (2/21)**  
이미지를 둘러본 바 분석 대상이 되는 객체(인물)의 위치가 중간 즈음에 고정되어있었습니다. 같은 배경을 가진 다른 인물들의 사진을 다수 발견하였습니다 (noise).
- **Mislabeling (2/22)**  
팀원 별로 각자 파트를 나눠 데이터 전수 검사를 진행하였습니다. 일부 식별이 되지 않는 경우를 제외하고 성별과 마스크 착용여부등을 다시 labeling 해주었습니다.
- **RGB 채널별 통계값 (2/23)**  
In computer vision, it is recommended to normalize image pixel values relative to the dataset mean and standard deviation. This helps to get consistent results when applying a model to new images and can also be useful for transfer learning. In practice, computing such statistics can be non-trivial. Pytorch dataloader provides a provides a method to do so. Refer to the [website](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html) for detailed procedures.  
전처리 과정 속 image normalization에서 주어진 baseline코드는 일부 데이터의 RGB 채널별 통계값을 구해 전처리 과정에서 normalize를 진행하였으나, 추후에 이를 전체 training dataset의 통계값을 구해 그 값을 hard coding해주는 방법으로 바꿔주었습니다. 이편이 일부 데이터를 보는 것보다 더 정확하고, 또 매번 계산해 줄 필요가 없어 더 효과적이라 생각했습니다.

## Dataset

- **Split by profile (2/24)**  
간단한 기학습 모델을 사용해 돌려본 결과 train/valid dataset 내부의 결과는 좋게 나오는 반면 실제 test dataset을 사용했을 때는 저조한 성적을 보였습니다. 이는 동일 인물의 사진이 여러 장 있음에도 이를 묶지 않고 random하게 train과 validation set을 나누었기 때문이었습니다. 처음에 데이터를 불러오는 과정에서 각 인물의 profile별로 dataset을 나누는 작업을 해주었습니다.
- **전처리 & 증식 (2/25-28)**  
EDA 결과를 토대로 이미지 전처리를 해줄 방법을 모색했습니다. torchvision.transform 보다 albumentation이 다양성과 속도 면에서 앞선다는 정보에 albumentation을 사용하였습니다. 기존의 baseline 코드는 torchvision을 사용하여 작성되었기에 이 [웹사이트](https://albumentations.ai/docs/examples/migrating_from_torchvision_to_albumentations/)를 참조하여 수정해주었습니다.  
*CenterCrop*. 주어진 데이터 내에서 인물이 대부분 중앙에 위치하였기에 CenterCrop을 적용시켜주었습니다. 보다 나은 성능을 위해 인물의 얼굴 위치를 잡아 이를 토대로 Crop 또는 bounding box를 잡아주는 방법을 모색하였으나 적용하지 못하였습니다. `cv2.CascadeClassifier` 를 적용시켜 보았으나 얼굴 위치를 잡아내는 성능면에서 많이 떨어진다고 판단하였고, 시간 상의 문제로 추가로 다른 방법을 찾아보지 못하였습니다.  
*Normalize & Resize*. 앞서 계산된 RGB 통계값을 사용하여 Normalize와 model에 사용되기 적합한 사이즈로 Resize해주었습니다.  
*ShiftScaleRotate, RandomBrightnessContrast, FancyPCA/GaussNoise*. 이번 대회의 경우 주어진 training dataset과 평가에 사용될 test dataset의 형식이 비슷하다는 것을 알기 때문에 증식 단계에서 많은 변화를 주지는 않았습니다. training dataset을 바탕으로 한 EDA내에서는 각 이미지 별로 조명, 밝기, 또는 위치나 각도 면에서 어느 정도 변화가 있었으며 이를 토대로 위와 같은 augmentation methods 들을 선택하였습니다. ColorJitter나 다른 augmentation 방법들도 여럿 염두해 두었으나 노년층 데이터의 흰머리, 주름과 같은 특징을 방해할 수 도 있다 생각해 적용 시키지 않았습니다.  
*preprocessing vs. augmentation*. preprocessing performs transforms ‘inplace’ whereas augmentation is transforming the data to create more samples (usually preventing overfitting). Augmentation의 경우 학습할 때 각 epoch마다 randomness를 부여해서 학습을 더 좋게 하는 것이다.
- **WeightedRandomSampler (2/25)**  
In each epoch, it selects the sample based on WeightedRandomSampler (less frequent data has higher change(weight) of being selected). Note that WeightedRandomSampler takes as input a list of weight where it contains a weight for entire training dataset.  
*데이터 불균형*. 현실의 과제에서 데이터 불균형은 매우 흔하게 나타납니다. 주요한 요인은 데이터 자체가 본질적으로 불균형 하기 때문입니다. 이러한 불균형의 가장 큰 문제점은 모델은 결국 학습용 데이터의 분포 (불균형)를 학습한다는 점이 있습니다. 그로 인해 제안 된 다양한 학습법들을 몇 적용시켰습니다. ([참고1](https://blog.mathpresso.com/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95%EC%97%90-%EB%8C%80%EC%9D%91%ED%95%98%EA%B8%B0-1-52af6aaebbf3), [참고2](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/))  
  1. 부족한 데이터를 반복적으로 노출 시키는 방법  
→ 예시 WeightedRandomSampler. 데이터의 양 측면에서는 효과적일 수 있으나, 데이터의 다양성을 향상 시키지는 못한다.
  2. 가상의 새로운 데이터를 생성하여 학습하는 방법  
→ 예시 SMOTE, GAN. SMOTE의 경우 과거에 주로 사용되었던 방법이다. 
  3. 정의된 목적 함수에 가중치를 다르게 주는 것  
→ 예시 WeightedCrossEntropy Loss. mini batch를 활용해서 학습이 진행되는 상황에서 이 방식은 위에서 소개한 데이터를 반복적으로 노출 시키는 방식과 이론적으로 크게 다르지 않다.
- **Downsample Mask (2/27)**  
정훈님께서 진행하신 실험을 바탕으로 적용하였습니다. 사실상 마스크를 쓴 이미지 5장이 별다른 차이 없이 중복된다고 생각하여 이 중 1장만 랜덤 하게 골라 사용하는 방식을 적용 후 크게 성능이 올라갔습니다.
- **Stratified k-folds (3/2)**  
기존의 k-fold cross validation은 전체 데이터셋이 k set으로 나뉘어 모두 한번씩 validation set이 된다는 점에서 많은 장점을 가지고 있습니다. 이를 통해 특정 데이터셋에 대한 과소적합을 방지할 수 있고, 더욱 일반화된 모델 생성이 가능합니다. 기존의 k-fold를 사용 시 각각의 set에서 label의 분포가 고르지 않을 수도 있는 현상이 있어 이점을 개선한 stratified k-fold를 적용시켜주었습니다.   
앞선 split by profile에서 처럼 동일 인물의 사진이 train과 validation set에 모두 포함 되 학습이 제대로 되지 않는 경우를 막아주기 위해 각 이미지가 아닌 인물별 profile에 대해 stratified k-fold를 적용시켜주었습니다 (이때 고유한 index를 유지하기 위해 shuffle을 해주지 않아야 합니다).
- **Future Works**  
*전처리된 이미지를 저장*. 전처리 이미지를 저장 후 사용 시에 불러오는 것이 속도 면에서 훨씬 효과적인 방법 이였으나 적용해 주지 못하였습니다. 추후에 다시 이와 같은 프로젝트 또는 대회에 임할 시 이를 염두 하여 적용 시키고 싶습니다.   
*Face Detection & 배경 날리기*. 앞서 말했듯이 cv2 라이브러리를 사용한 face detection을 시도하였으나 성능이 따라주지 않아 실패하였습니다. 마스터 세션에서 상위권에 위치한 팀원들이 시도 한 것처럼 추후에 다시 이와 같은 대회에 임할 시에 face detection과, EDA단계에서 발견했던 배경 속 noise handling을 위한 배경 날리기 또한 시도 해보고 싶습니다 (같은 배경 속 다른 인물 사진). 이러한 대회에서는 구현이 중요한 만큼 데이터 자체를 파악하는 것이 중요하다는 점을 배웠습니다.  
*Validation set ratio*. 다른 팀이 적용 했던 방법 중에 validation set ratio를 기존에 설정되어 있던 0.2가 아닌 0.1로 변경하였을 때 크게 성능이 올라갔다는 말을 들었습니다. 주어진 학습 데이터 사이즈가 크지 않았기에 이와 같은 성능을 냈던 것 같습니다.  
종합해 보자면 전반적으로 주어진 데이터셋에 대한 이해도가 부족했다고 생각합니다. “얼굴 이미지의 나이, 성별, 마스크 분류 문제”에 초점을 맞추는 것이 아닌 데이터셋의 특징 “적은 데이터셋, 각 클래스 별 불균형의 정도”에 초점을 맞추었다면 좀 더 나은 결과를 이루었겠음을 배웠습니다.  
*Change Labels*. 대회가 끝난 후 다른 팀이 적용한 방법 중 학습 데이터 레이블을 다음과 같이 바꿔줄 수 있었다고 깨달았습니다.  
  1. 비교적 적은 “60대 이상” 클래스의 분포를 늘려주기 위해 이를 “50대 중후반 이상” 으로 바꿔주기.
  2. 명확한 연령대 학습을 위해 각 연령대 class의 boundary에 있는 데이터 지워주기.  

이와 같이 원활한 학습을 위해 데이터셋에 적절한 변화가 필요하다는 것을 배웠고, 또 기회가 있다면 이번에 배운 교훈들을 토대로 추후에 적용 시켜보고 싶습니다.  


## Model

- **pretrained models (2/24-28)**  
ImageNet 등의 데이터를 통해 기학습된 모델 사용이 가능하였기에 모델 조사를 진행하였습니다. 모델의 구조 자체에는 크게 변화를 주지 않을 것이라 생각해 pytorch 공식 홈페이지에 있는 [기학습 모델 사용법](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)을 인용하였습니다. 초반에 진행하였던 실험들은 가장 간단한 구조라고 생각되는 ResNet18을 사용하였습니다.  
기학습된 모델의 마지막 dense layer을 새로 지정 해줄 때 Xavier initialization을 초반에는 해주었으나, 추후에 성능에 크게 영향이 없다고 생각해 제외하였습니다 ([참고](https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52)).  
*Xavier initialization of dense layer.* proposed by Xavier Glorot in 2010. The main objective of it to initialize weights such that the activations have a mean of 0 and standard deviation of 1. This concept relates to exploding and vanishing gradient problem.  
*fine-tuning*. pre-trained 된 모델로 시작하여 새로운 task에 대한 model의 모든 parameter을 업데이트. 본질적으로 전체 model을 retraining한다.  
*feature extraction*. pre-trained된 모델로 시작하여 prediction을 도출하는 마지막 레이어의 weight만 업데이트 한다. ([참고](https://better-tomorrow.tistory.com/entry/TorchVision-model-funetuning))  
“An assessment of the 1.4 million images included in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset found that **17% of the images contain faces**, despite the fact that only three of 1,000 categories in the dataset mention people.”  
위와 같이 ImageNet과 주어진 task dataset의 종류가 다르다고 생각해 ImageNet 모델 사용 시 전체 레이어를 fine-tuning 해주는 것이 낫다고 생각하였습니다.  
ImageNet dataset이 아닌 얼굴 이미지 데이터로 학습된 모델이 있는지 조사를 하던 도중 [VGGFace모델을 발견](https://discuss.pytorch.org/t/pretrained-vgg-face-model/9383)하였습니다. 내부적으로 VGG16과 같은 구조라는 것을 참고해 해당 [웹페이지](https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth)로부터 weight을 load해 사용해보았습니다.  VGGFace의 경우 fine-tuning의 정도를 다르게 하여 그 성능을 비교해 보았습니다.  
- **각 클래스 별로 모델 학습 시키기 (2/25)**  
모델의 구조를 설계할 때 더욱 간단한 (simplified) task를 주는 것이 성능 개선에 좋을 것이라 판단하였습니다. 하여, 주어진 이미지를 식별하는 나이, 성별, 마스크 착용상태의 여부를 18개 클래스로 구분하는 단일 모델이 아닌 각각의 task로 가진 3개의 다른 모델로 학습하도록 변환시켜 주었습니다. 추론 단계에서 각각의 모델의 예측 결과를 알맞게 encoding하는 식으로 적용하였습니다.  
- **Future Works**  
*Transformer*. Pytorch에 내장 되어있는 pretrained model들 밖에 사용을 못해 보았던게 아쉽습니다. ViT 또는 Swin Transformer을 구현 후 pretrained weight을 불러와 적용시켰을 때 성능이 좋았다고 들어 이를 적용시켜보고 싶습니다.  
*Pytorch Lightning*. Pytorch Lightning은 복잡한 양의 딥러닝 코드 작업을 추상화 시켜줄 수 있도록 굉장히 잘 구조화 되어있고 다양한 학습 방법에 적용이 가능하다고 들었습니다. 이에 대해 조금더 조사하고 추후에 적용시켜 보고 싶습니다.  

## Metric (Loss Function & Optimization)

- **WeightedCrossEntropy Loss (2/26)**  
데이터 불균형이 심할 때 사용되는 방법 중에 하나라고 하여 구현을 해 보았습니다.  
Weighted Cross Entropy works in the training phase. It is used to solve the problem that the accuracy of the model overfitting on the test set due to the imbalance of the convergence speed of the loss function decrease.  
- **Loss, Optimizer (3/1-3)**  
정빈님께서 WeightedCrossEntropy를 대조군으로 두고 focal loss, label smoothing loss, f1 loss와 비교 진행하였습니다. 이 과정에서 f1 loss는 실제로 gradient 계산이 가능하게 끔 구현해준 것이 아니라 정확한 비교가 불가능 하였습니다.   
상우님께서 SGD를 대조군으로 두고 Adam과 AdamW를 비교 진행하였습니다.  
- **Learning Rate Scheduler (3/1-3)**  
*Learning Rate Scheduler*. 처음부터 끝까지 같은 learning rate를 사용할 수도 있지만, 학습 과정에서 learning rate를 조정하는 learning rate scheduler를 사용할 수도 있다. 처음엔 큰 learning rate(보폭)으로 빠르게 optimize를 하고 최적 값에 가까워질수록 learning rate(보폭)를 줄여 미세조정을 하는 것이 학습이 잘된다고 알려져 있다. ([참고](https://sanghyu.tistory.com/113))  
정빈님께서 StepLR을 대조군으로 두고 ExponentialLR을 비교 진행하였습니다.  
- **Hyperparameter Tuning (3/3)**  
진우님께서 wandb sweep을 사용하여 hyperparameter tuning을 진행하였습니다. stratified k-folds를 적용한 상태에서는 구현이 어려워 그 이전의 코드로 진행하셨기에 결과를 적용시키지는 않았습니다.  
- **Future Works**  
*f1 loss*. 미분이 가능하지 않은 f1 loss를 평가 지표로 사용하기 위해 1. 임계값을 지정하여 f1점수를 최대화 하거나 2. f1점수를 손실 함수에 포함시키는 방법 등을 생각해볼 수 있다.  

## Training

- **Early stopping (2/25)**  
적합한 epoch빈도를 모르는 상황에서 overfitting을 막아주기 좋을 것이라 생각해 구현하였습니다.  
- **Future Works**  
*Checkpoints & Resume*. 기존의 [pytorch template](https://github.com/victoresque/pytorch-template)에 있는 유틸 기능 중 하나로 모델 학습 도중 checkpoint를 저장해 이를 사용해 추후에 모델을 마저 학습 시킬 수 있는 기능을 구현하여 사용하는 것이 더욱 효과적 이었을거라 생각합니다.   
*Gradient Cam*. CNN의 내부를 볼 수 있는 도구인 CAM의 일반화 버전으로 convolution layer을 fine tuning해서 CNN에서 이미지에 어떤 부분에 더 집중하고 있는지를 확인할 수 있는 라이브러리 입니다. 이와 같은 라이브러리를 통해 단순히 성적을 보고 잘 작동하는지 아는 여부를 떠나 정말, 왜, 어떠한 방법으로 잘 되고 있는지를 확인해 보고 싶습니다. ([참고1](https://jsideas.net/grad_cam/), [참고2](https://github.com/jacobgil/pytorch-grad-cam))  

## Evaluation

- **Confusion Matrix (2/28)**  
정훈님께서 학습된 모델의 성능을 제출을 하지 않아도 비교 분석할 수 있는 시각화 모듈을 구현하였습니다.  
- **Ensemble, Voting (3/4)**  
별다른 과정 없이 서로 다른 모델들을 결합해 개선된 성능을 보일 수 있는 방법이라고 생각해 soft voting과 hard voting을 구현하였습니다. 성능면에서는 soft voting방법이 앞섰으나 단일모델 효과가 더 좋아 단일 모델을 최종적으로 제출하였습니다.  
- **Future Works**  
*Ensemble, Boosting*. 주어진 데이터 셋 사이즈가 적기는 하지만, 불균형이 심한 데이터셋에 좋다고 알려진 Adaboost등을 구현 및 적용해보고 싶습니다.  
