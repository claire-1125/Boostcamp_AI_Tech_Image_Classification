python train.py --epochs 60 --model PretrainedModels --model_param resnet18 false --weight_version 3 --name ResNet18_Ep60_Weightv3_AGE --label age
python train.py --epochs 60 --model PretrainedModels --model_param resnet18 false --weight_version 3 --name ResNet18_Ep60_Weightv3_GENDER --label gender
python train.py --epochs 60 --model PretrainedModels --model_param resnet18 false --weight_version 3 --name ResNet18_Ep60_Weightv3_MASK --label mask
python inference.py --model PretrainedModels --model_param resnet18 false --output_filename output-ResNet18_Ep60_AGM-20220302-ijkimmmy-v13.csv --label age gender mask --model_dir ./model/age/ResNet18_Ep60_Weightv3_AGE ./model/gender/ResNet18_Ep60_Weightv3_GENDER ./model/mask/ResNet18_Ep60_Weightv3_MASK
