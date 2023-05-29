# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --output_filename output_weightedFocal_weight0.csv --model_dir ./model/age/age_weightedFocal_weight0 ./model/gender/gender_weightedFocal_weight0 ./model/mask/mask_weightedFocal_weight0 --batch_size=250

# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --output_filename output_label_smoothing.csv --model_dir ./model/age/age_label_smoothing ./model/gender/gender_label_smoothing ./model/mask/mask_label_smoothing --batch_size=250

# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --output_filename output_labelSmoothing0.2_weight1.csv --model_dir ./model/age/age_labelSmoothing0.2_weight1 ./model/gender/gender_labelSmoothing0.2_weight1 ./model/mask/mask_labelSmoothing0.2_weight1 --batch_size=250

# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --output_filename output_weightedFocal_weight1.csv --model_dir ./model/age/age_weightedFocal_weight1 ./model/gender/gender_weightedFocal_weight1 ./model/mask/mask_weightedFocal_weight1 --batch_size=250

# python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --output_filename output_labelSmoothing0.2_weight3_5fold.csv --model_dir ./model/age/age_labelSmoothing0.2_weight3_5fold ./model/gender/gender_labelSmoothing0.2_weight3_5fold ./model/mask/mask_labelSmoothing0.2_weight3_5fold --batch_size=250

python inference.py --model PretrainedModels --model_param resnet false --label age gender mask --output_filename ResNet50_output_labelSmoothing0.2_weight3_5fold.csv --model_dir ./model/age/ResNet50_age_labelSmoothing0.2_weight3_5fold ./model/gender/ResNet50_gender_labelSmoothing0.2_weight3_5fold ./model/mask/ResNet50_mask_labelSmoothing0.2_weight3_5fold --batch_size=250
