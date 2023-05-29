# 2022-02-25
# separate label & implemented WeightedRandomSample
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_AGE --label age
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_MASK --label mask
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation BaseAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_BaseAug_GENDER --label gender

# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug_AGE --label age
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug_MASK --label mask
# python train.py --epochs 20 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAug_GENDER --label gender

# python train.py --epochs 30 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Ep20_SplitProf_CustAugv1

# python inference.py --model VGGFace --model_dir ./model/multi/VGGFace_Ep20_SplitProf_CustAugv1
# python inference.py --model VGGFace --label age gender mask --model_dir ./model/age/VGGFace_Ep20_SplitProf_CustAug_AGE ./model/gender/VGGFace_Ep20_SplitProf_CustAug_GENDER ./model/mask/VGGFace_Ep20_SplitProf_CustAug_MASK

# python train.py --epochs 50 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer Adam --name VGGFace_Ep20_SplitProf_CustAugv1_Adam
# python train.py --epochs 50 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep20_SplitProf_CustAugv1_WeightedCE_SGD

# 2022.02.26
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnNormSample_SGD
# python inference.py --model PretrainedModels --model_param resnet false --model_dir ./model/multi/ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet true --optimizer SGD --name ResNet_Feature_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Freeze6_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_fcxavier_Ep60_SplitProf_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_MASK --label mask
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --name ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_GENDER --label gender

# 2022.02.27
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param alexnet false --optimizer SGD --name Alexnet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param squeezenet false --optimizer SGD --name Squeezenet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param densenet false --optimizer SGD --name Densenet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param inception false --optimizer SGD --name Inception_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --name VGGFace_Feature_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER --label gender
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_MASK --label mask

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_MASK --label mask
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --optimizer SGD --name VGGFace_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER --label gender

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER --label gender
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --model_dir ./model/age/ResNet_Ep120_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_AGE ./model/gender/ResNet_Ep120_patience30_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev2_SGD_GENDER ./model/mask/ResNet_Ep60_SplitProf_Downsample_CustAugv1_WeightedCEnSample_SGD_MASK

# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_AGE --label age
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_GENDER --label gender
# python train.py --epochs 120 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_MASK --label mask
# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --model_dir ./model/age/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_AGE ./model/gender/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_GENDER ./model/mask/ResNet_Ep120_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev1_SGD_MASK

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_MASK --label mask

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param true --optimizer SGD --name VGGFace_Feature_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_AGE --label age

# 2022.02.28 (fix augmentation)
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model PretrainedModels --model_param resnet false --optimizer SGD --name ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK --label mask

# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --model_dir ./model/age/ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_AGE ./model/gender/ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_GENDER ./model/mask/ResNet_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev3_SGD_MASK
# python inference.py --model VGGFace --model_param resnet false --label age gender mask --model_dir ./model/age/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_AGE ./model/gender/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_GENDER ./model/mask/VGGFace_Ep60_patience15_SplitProf_Downsample_CustAugv1_WeightedCEnSamplev0_SGD_MASK

# compare weight version diff
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv3_MASK --label mask

# python inference.py --model VGGFace --model_param false --label age gender mask --output_filename output-VGGFace_Ep60_Weightv3_AGM-20220301-ijkimmmy.csv --model_dir ./model/age/VGGFace_Ep60_Weightv3_AGE ./model/gender/VGGFace_Ep60_Weightv3_GENDER ./model/mask/VGGFace_Ep60_Weightv3_MASK

# efficientnet try1
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 1 --name VGGFace_Ep60_Weightv1_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 1 --name VGGFace_Ep60_Weightv1_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 1 --name VGGFace_Ep60_Weightv1_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_MASK --label mask

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 2 --name VGGFace_Ep60_Weightv2_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 2 --name VGGFace_Ep60_Weightv2_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 2 --name VGGFace_Ep60_Weightv2_MASK --label mask

# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 0 --name VGGFace_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 0 --name VGGFace_Ep60_Weightv0_GENDER --label gender
# python train.py --epochs 60 --dataset MaskSplitByProfileDataset --augmentation CustomAugmentation --model VGGFace --model_param false --optimizer SGD --weight_version 0 --name VGGFace_Ep60_Weightv0_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --name Vgg_Ep60_Weightv3_MASK --label mask

# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 3 --name Vgg_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 2 --name Vgg_Ep60_Weightv2_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 1 --name Vgg_Ep60_Weightv1_AGE --label age

# # python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_GENDER --label gender
# # python train.py --epochs 60 --model PretrainedModels --model_param vgg false --optimizer SGD --weight_version 4 --name Vgg_Ep60_Weightv4_MASK --label mask

# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 3 --name VGGFace_Feature_Ep60_Weightv3_AGE --label age
# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 3 --name VGGFace_Feature_Ep60_Weightv3_GENDER --label gender
# python train.py --epochs 60 --model VGGFace --model_param true --optimizer SGD --weight_version 3 --name VGGFace_Feature_Ep60_Weightv3_MASK --label mask

############################################################

# Effb3_Ep60_Weightv0_AGE
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_AGE --label age
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_AGE --label gender
# python train.py --epochs 60 --model PretrainedModels --model_param efficientnet-b3 false --optimizer SGD --name Effb3_Ep60_Weightv0_AGE --label mask

# python inference.py --output_filename Effb3_Ep60_Weightv0_AGE --model_dir ./model/age/Effb3_Ep60_Weightv0_AGE2 ./model/gender/Effb3_Ep60_Weightv0_AGE ./model/gender/Effb3_Ep60_Weightv0_AGE --model_param efficientnet-b3 false --label age gender mask --model PretrainedModels
# Effb3_Ep60_Weightv0__AGE

# VGGFace_Ep60_Weightv0_CustomAug
# python train.py --epochs 60 --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv0_CustomAug --augmentation CustomAugmentation --label age --weight_version 0 
# python train.py --epochs 60 --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv0_CustomAug --augmentation CustomAugmentation --label gender --weight_version 0 
# python train.py --epochs 60 --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv0_CustomAug --augmentation CustomAugmentation --label mask --weight_version 0 

# python inference.py --model VGGFace --model_param false  --output_filename VGGFace_Ep60_Weightv0_CustomAug --model_dir ./model/age/VGGFace_Ep60_Weightv0_CustomAug ./model/gender/VGGFace_Ep60_Weightv0_CustomAug ./model/mask/VGGFace_Ep60_Weightv0_CustomAug --label age gender mask 

# VGGFace_Ep60_Weightv0_CustomAug_without_oneof
# python train.py --epochs 60 --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv0_CustomAug_without_oneof --augmentation CustomAug_without_oneof --label age --weight_version 0 
# python train.py --epochs 60 --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv0_CustomAug_without_oneof --augmentation CustomAug_without_oneof --label gender --weight_version 0 
# python train.py --epochs 60 --model VGGFace --model_param false --optimizer SGD --name VGGFace_Ep60_Weightv0_CustomAug_without_oneof --augmentation CustomAug_without_oneof --label mask --weight_version 0 

# python inference.py --model VGGFace --model_param false  --output_filename VGGFace_Ep60_Weightv0_CustomAug_without_oneof --model_dir ./model/age/VGGFace_Ep60_Weightv0_CustomAug_without_oneof ./model/gender/VGGFace_Ep60_Weightv0_CustomAug_without_oneof ./model/mask/VGGFace_Ep60_Weightv0_CustomAug_without_oneof --label age gender mask 


# ResNet50_Ep60_CustomAug
python train.py --epochs 60 --model ResNet50 --optimizer SGD --name ResNet50_Ep60_CustomAug --augmentation CustomAugmentation --label age --weight_version 3     --k_folds 5 
python train.py --epochs 60 --model ResNet50 --optimizer SGD --name ResNet50_Ep60_CustomAug --augmentation CustomAugmentation --label gender --weight_version 3  --k_folds 5 
# python train.py --epochs 60 --model ResNet50 --optimizer SGD --name ResNet50_Ep60_CustomAug --augmentation CustomAugmentation --label mask --weight_version 3    --k_folds 5 

# python inference.py --model ResNet50 --output_filename ResNet50_Ep60_CustomAug --model_dir ./model/age/ResNet50_Ep60_CustomAug ./model/gender/ResNet50_Ep60_CustomAug ./model/mask/ResNet50_Ep60_CustomAug --label age gender mask 

# kfold 적용 안 된 상태
# train.py의 132줄의 sampler 주석처리햇으니까 다시 돌려놓기!!!!!!!!


# ResNet50_Ep60_CustomAug_W3_AdamW_
python train.py --epochs 60 --model ResNet50 --optimizer AdamW --name ResNet50_Ep60_CustomAug_W3_AdamW_ --augmentation CustomAugmentation --label age --weight_version 3    --k_folds 5 
python train.py --epochs 60 --model ResNet50 --optimizer AdamW --name ResNet50_Ep60_CustomAug_W3_AdamW_ --augmentation CustomAugmentation --label gender --weight_version 3 --k_folds 5 
# python train.py --epochs 60 --model ResNet50 --optimizer AdamW --name ResNet50_Ep60_CustomAug_W3_AdamW_ --augmentation CustomAugmentation --label mask --weight_version 3   --k_folds 5 

# python inference.py --model ResNet50 --output_filename ResNet50_Ep60_CustomAug_SamplingDataset --model_dir ./model/age/ResNet50_Ep60_CustomAug_SamplingDataset ./model/gender/ResNet50_Ep60_CustomAug_SamplingDataset ./model/mask/ResNet50_Ep60_CustomAug_SamplingDataset --label age gender mask 

# ResNet50_Ep60_CustomAug_W3_SGD_Decay
python train.py --epochs 60 --model ResNet50 --lr_decay_step 10 --optimizer SGD --name ResNet50_Ep60_CustomAug_W3_SGD_Decay --augmentation CustomAugmentation --label age --weight_version 3    --k_folds 5 

# ResNet50_Ep60_CustomAug_annealing
python train.py --epochs 60 --model ResNet50 --optimizer SGD --name ResNet50_Ep60_CustomAug_annealing --augmentation CustomAugmentation --label age --weight_version 3    --k_folds 5 
