import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

import model.model as model_model
from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, model_param, device):
    model_cls = getattr(model_model, args.model)
    model = model_cls(
        num_classes=num_classes,
        **model_param
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    import numpy as np
    
    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    
    preds = []
    lst_labels = ['age', 'gender', 'mask']
    model_param_module = getattr(import_module("train"), 'parse_model_param')
    model_param = model_param_module(args.model_param, 'Pretrained' in args.model)
    
    if 'multi' in args.label:
        preds = inference_model(loader, model_dir[0], model_param, 18)
    elif all(item in args.label for item in lst_labels):
        index = {name: args.label.index(name) for name in lst_labels}
        pred_age = np.array(inference_model(loader, model_dir[index['age']], model_param, 3))
        pred_gender = np.array(inference_model(loader, model_dir[index['gender']], model_param, 2))
        pred_mask = np.array(inference_model(loader, model_dir[index['mask']], model_param, 3))
        preds = pred_mask * 6 + pred_gender * 3 + pred_age
    # else:
    #     raise ValueError(f"Must pass either 1 or 3 models.. passed {len(model_dir)}")
    
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, args.output_filename), index=False)
    print(f'Inference Done!')


@torch.no_grad()
def inference_model(loader, model_dir, model_param, num_classes) -> list:
    # num_classes = MaskBaseDataset.num_classes  # 18
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = load_model(model_dir, num_classes, model_param, device).to(device)
    model.eval()

    print(f"Calculating inference results for {model_dir}..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--model_param', nargs='+', default='resnet false', help='model type (default: BaseModel)')
    parser.add_argument('--label', nargs='+', default='multi')
    parser.add_argument('--output_filename', type=str, default='output.csv')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', nargs='+', default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    print(model_dir, "모델 경로")
    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
