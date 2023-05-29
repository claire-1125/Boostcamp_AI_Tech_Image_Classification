import argparse  # 스크립트 호출 시 인자값을 조정하기 위함.
import glob  # 파일 경로 리스트 사용하기 위함.
import json
import multiprocessing
import os
from pkgutil import get_data
from cv2 import transform
from dotenv import load_dotenv
import random
import re  # regular expression 모듈
from importlib import import_module  # 동적 import를 위함.
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR  # learning rate를 유동적으로 조절
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # tensorboard 기반으로 logging
import matplotlib.pyplot as plt
import wandb

# import model as model_model
from dataset import MaskBaseDataset
from loss import create_criterion
from util import EarlyStopping


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)  # "./data/exp"이라는 파일 경로 객체
    # 1. 파일 경로가 존재하고 exist_ok가 True인 경우
    # 2. 파일 경로가 존재하지 않거나
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")  # 조건에 맞는 파일명을 리스트로 변환
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]  # exp0, exp1 같은 형태의 파일명이 존재하는지 체크
        i = [int(m.groups()[0]) for m in matches if m]  # exp 뒤에 붙는 숫자를 리스트에 저장
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def parse_model_param(params:str, pretrained: bool) -> dict: 
    """
    Args
        params : args.model_param (default : resnet false)
        pretrained : args.model이 VGGFace나 PretrainedModels 클래스인지의 여부
    """

    model_param = {}
    if pretrained:
        model_names = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
        for param in params:  # resnet false
            if param.lower() in model_names:
                model_param['model_name'] = param.lower()
            if param.lower() == 'true':
                model_param['feature_extract'] = True
            elif param.lower() == 'false':
                model_param['feature_extract'] = False
    return model_param


def get_dataloder(dataset, train_trfm, val_trfm, args):
    """_summary_
    enumerate through fold indices, split dataset into train and validation set, init transform for each
    returns list of (train_dataloader, valid_dataloader)
    Args:
        dataset (MaskSplitByProfileDataset)
            : dataset containing stratified kfold information with regards to the profile of the image
        train_trfm
            : train transform
        val_trfm
            : validation transform
        args
            : arguments passed in
    """
    assert dataset.n_fold == args.k_folds

    dataloaders = []    
    for index in range(args.k_folds):
        dataset.set_indices(index)
        train_set, val_set = dataset.split_dataset()
        train_set.set_transform(train_trfm)
        val_set.set_transform(val_trfm)
        sampler = dataset.get_weighted_sampler(args.weight_version) # WeightedRandomSampler
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            # shuffle=True,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            sampler=sampler,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count()//2,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        dataloaders.append((train_loader, val_loader))
    return dataloaders


def train(data_dir, model_dir, args):
    """
    Args:
        data_dir = $SM_CHANNEL_TRAIN$ = "/opt/ml/input/data/train/images"
        model_dir = $SM_MODEL_DIR$ = "./model/레이블명(e.g.multi)"
    """
    
    seed_everything(args.seed)  # random seed가 필요한 곳에 seed값 세팅

    # 인자 : "./model/exp"
    save_dir = increment_path(os.path.join(model_dir, args.name))

     # CPU or GPU settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 전체 dataset 세팅
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskSplitByProfileDataset
    dataset = dataset_module(
        data_dir=data_dir,
        label=args.label,
        n_fold=args.k_folds
    )
    num_classes = dataset.num_classes 

    # 전체 dataset을 train set vs. validataion set으로 split
    # train_set, val_set = dataset.split_dataset()
    
    # train dataset에 augmentation 적용
    # transform_module = getattr(import_module("dataset"), args.augmentation)  # default : CustomAugmentation
    # transform = transform_module(
    #     resize=args.resize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    # train_set.set_transform(transform)
    
    # # validation dataset에 대해 preprocessing
    # transform_module = getattr(import_module("dataset"), 'BaseAugmentation') 
    # transform = transform_module(
    #     resize=args.resize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    # val_set.set_transform(transform)
        

    train_trfm_module = getattr(import_module("dataset"), args.augmentation)  # default : CustomAugmentation
    train_trfm = train_trfm_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std
    )
    
    # validation dataset에 대해 preprocessing
    val_trfm_module = getattr(import_module("dataset"), 'BaseAugmentation') 
    val_trfm = val_trfm_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std
    )

    dataloaders = get_dataloder(dataset, train_trfm, val_trfm, args)


    # train dataset에 대해 class weight 넣은 sampler 생성
    """
    tmi : 여기서는 전체 dataset에 대해 해당 함수를 호출했지만
        dataset.py 내부 구현 상 실제로는 train 데이터에 대해서만 
        weightedRandomSampler를 생성한다.
    """
    # sampler = dataset.get_weighted_sampler(args.weight_version) # WeightedRandomSampler


    # data-loader 세팅
    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     num_workers=multiprocessing.cpu_count()//2,
    #     # shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    #     sampler=sampler,
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=args.valid_batch_size,
    #     num_workers=multiprocessing.cpu_count()//2,
    #     shuffle=False,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )


    # model 세팅
    pretrained = args.model in ['VGGFace', 'PretrainedModels']  # 인자값으로 VGGFace나 PretrainedModel을 주었는지의 여부
    model_param = parse_model_param(args.model_param, pretrained)  # {'model_name':resnet,'feature_extract':False}
    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=num_classes, **model_param).to(device)  # 모델 객체 생성
    model = torch.nn.DataParallel(model) # 데이터 병렬 처리


    # loss function & optimizer 세팅
    f1_score = create_criterion('f1', **{'classes':dataset.num_classes})


    # 각 loss function 타입에 따른 인스턴스 생성
    if args.criterion == "cross_entropy":
        # weighted cross entropy loss
        class_weight = dataset.compute_class_weight()
        criterion = create_criterion(args.criterion, **{'weight':class_weight})
    elif args.criterion == "label_smoothing":
        criterion = create_criterion(args.criterion, **{'classes':num_classes})
    elif args.criterion == "focal":
        # weighted focal loss
        class_weight = dataset.compute_class_weight()
        criterion = create_criterion(args.criterion, **{'weight':class_weight})
    elif args.criterion == "f1":
        criterion = create_criterion(args.criterion, **{'classes':num_classes})
    elif args.criterion == "arcface":
        criterion = create_criterion(args.criterion, **{'easy_margin':True})


    # criterion = create_criterion(args.criterion)  # default: cross_entropy
    

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    
    # tensorboard의 SummaryWriter 인스턴스를 만들기 이전에 init해야 한다!
    wandb.init(project="Image_Classification", entity="claire_1125", sync_tensorboard=True)
    wandb.config.update(args) # argument로 넘겨준 configuration 값들을 wandb의 config 변수로 전달한다.


    logger = SummaryWriter(log_dir=save_dir)  # log_dir : tensorboard의 log 기록이 저장될 공간 (즉. "./model/exp숫자")
    
    # 현재 지정된 configuration값(argument 지정값들)들을 './model/exp숫자' 디렉토리 내에 config.json으로 저장한다.
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:  # 파일스트림 객체 자동 여닫음
        json.dump(vars(args), f, ensure_ascii=False, indent=4)  # 실제 json 파일로 저장
    
    # early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.0)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    for epoch in range(args.epochs):
        if early_stopping.stop:  # break outer loop as well
            break

        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for fold_idx, (train_loader, val_loader) in enumerate(dataloaders):
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Fold[{fold_idx}]Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                    wandb.log({
                        "Epoch": epoch * len(train_loader) + idx,
                        "Train/loss": train_loss, 
                        "Train/accuracy": train_acc
                    })

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items, val_acc_items = [], []
                figure = None
                out_lst, pred_lst = [], []
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    out_lst.append(outs.cpu().data)
                    pred_lst.append(preds.cpu())

                    loss_item = criterion(outs, labels).item() 
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )
            
                out_lst = torch.cat(out_lst)
                pred_lst = torch.cat(pred_lst)
                val_f1 = f1_score(out_lst, pred_lst)
                best_val_f1 = max(best_val_f1, val_f1)

                val_loss = np.sum(val_loss_items) / len(val_loader)
                # val_acc = np.sum(val_acc_items) / len(val_set)
                # best_val_loss = min(best_val_loss, val_loss)
            
                early_stopping(val_loss)
                if early_stopping.stop:
                    print("Early Stopping")
                    break
            
                if val_loss > best_val_acc:
                    print(f"New best model for val loss : {val_loss:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_loss = val_loss
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] f1 : {val_f1:4.2%}, loss: {val_loss:4.2} || "
                    f"best f1 : {best_val_f1:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/f1", val_f1, epoch)
                logger.add_scalar("Val/loss", val_loss, epoch)
                # logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                wandb.log({
                    "Epoch": epoch, 
                    "Val/f1": val_f1,
                    "Val/loss": val_loss, 
                    # "Val/accuracy": val_acc, 
                    "results": figure
                    })
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 인자값을 받을 객체
    
    load_dotenv(verbose=True)

    ## 입력받을 인자값 등록
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train (default: 60)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: CustomAugmentation)')
    ########### image size 고려해보기
    # nargs : 해당 옵션에 대한 인자 개수 ("+":1개 이상의 값을 전부 읽어들인다.)
    parser.add_argument("--resize", nargs="+", type=list, default=[224, 224], help='resize size for image when training') 
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=200, help='input batch size for validing (default: 200)')
    parser.add_argument('--k_folds', type=int, default=5, help='number of splits using k-fold (default: 5)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    
    # 추가
    parser.add_argument('--model_param', nargs='+', default='resnet false', help='model parameter (default: resNet false)')
    
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')    
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    # 추가
    parser.add_argument('--label', type=str, default='multi', help='label of the data (default: multi)')

    # 추가
    parser.add_argument('--weight_version', type=int, default=0, help='implementation version of WeightedRandomSampler (default: 0)')


    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))  # 환경변수명, (값이 없을 경우) 그 환경변수명의 default값
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'))

    # 입력받은 인자값을 저장 (type: namespace)
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir + '/' + args.label

    train(data_dir, model_dir, args)
    
