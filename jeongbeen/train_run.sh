
# weighted focal loss
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label mask --name mask_weightedFocal_weight0 --weight_version 0
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label gender --name gender_weightedFocal_weight0 --weight_version 0
# /python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label age --name age_weightedFocal_weight0 --weight_version 0

# label smoothing
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label mask --name mask_label_smoothing
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label gender --name gender_label_smoothing
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label age --name age_label_smoothing

# renew label smoothing (smoothing=0.2)
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label mask --name mask_labelSmoothing_weight1 --weight_version 1
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label gender --name gender_labelSmoothing_weight1 --weight_version 1
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label age --name age_labelSmoothing_weight1 --weight_version 1

# weighted focal
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label mask --name mask_weightedFocal_weight1 --weight_version 1
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label gender --name gender_weightedFocal_weight1 --weight_version 1
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion focal --label age --name age_weightedFocal_weight1 --weight_version 1

# f1 loss
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion f1 --label mask --name mask_f1_weight1 --weight_version 1
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion f1 --label gender --name gender_f1_weight1 --weight_version 1
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion f1 --label age --name age_f1_weight1 --weight_version 1

# label smoothing (smoothing=0.2) weight version:3
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label mask --name mask_label_smoothing_weight3 --weight_version 3 --k_folds 5
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label gender --name gender_label_smoothing_weight3 --weight_version 3 --k_folds 5
# python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label age --name age_label_smoothing_weight3 --weight_version 3 --k_folds 5

# label smoothing (smoothing=0.2) weight version:3
python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label mask --name ResNet50_mask_labelSmoothing0.2_weight3_5fold --weight_version 3 --k_folds 5
python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label gender --name ResNet50_gender_labelSmoothing0.2_weight3_5fold --weight_version 3 --k_folds 5
python train.py --epochs 60 --model PretrainedModels --model_param resnet false --criterion label_smoothing --label age --name ResNet50_age_labelSmoothing0.2_weight3_5fold --weight_version 3 --k_folds 5
