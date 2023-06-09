##################################### baseline code #####################################

from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseModel(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

###########################################################################################
# pretrained VGGFace (base VGG16)
# https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth
###########################################################################################
class VGGFace(nn.Module):
    def __init__(self, num_classes, feature_extract: bool=False, dict_weight: OrderedDict=None, *args, **kwargs):
        super().__init__()
        self.model = models.vgg16()
        self.num_classes = num_classes
        self.dict_weight = dict_weight
        self.feature_extract = feature_extract
        
        self.init_weights()
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.model(x)
    
    def print(self):
        import numpy as np
        
        np.set_printoptions(precision=3)
        n_param = 0
        for p_idx,(param_name,param) in enumerate(self.model.named_parameters()):
            if param.requires_grad:
                param_numpy = param.detach().cpu().numpy() # to numpy array 
                n_param += len(param_numpy.reshape(-1))
                print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
                print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
        print ("Total number of parameters:[%s]."%(format(n_param,',d')))
    
    def init_weights(self):
        from torch.utils import model_zoo
        
        # load weights and update label for the loaded state dict (weights)
        if self.dict_weight is None:
            weight_url = 'https://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/vgg_face_dag.pth'
            self.dict_weight = model_zoo.load_url(weight_url)
        vgg_labels = lst = [name for name, _ in self.model.named_parameters() if name.split(sep='.')[0]=='features']
        vggface_weights = list(self.dict_weight.items())
        vggface_weights = [(vgg_labels[idx], vggface_weights[idx][1]) for idx in range(len(vgg_labels))]
        
        self.model.load_state_dict(dict(vggface_weights), strict=False) # strict=False.. otherwise it raises Key Error
        self.set_param_requires_grad()
        self.model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes)
        )
        
    def set_param_requires_grad(self):
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False


# Pretrained Models 
# https://tutorials.pytorch.kr/beginner/finetuning_torchvision_models_tutorial.html
class PretrainedModels(nn.Module):
    def __init__(self, num_classes, model_name: str='resnet18', feature_extract: bool=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.model = None
        self.input_size = None

        self.init_model()        

    def forward(self, x):
        return self.model(x)

    def init_model(self):
        # import math
        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.set_param_requires_grad()
            in_features = self.model.fc.in_features  # 512
            self.model.fc = torch.nn.Linear(in_features=in_features, out_features=self.num_classes)
            # torch.nn.init.xavier_uniform_(self.model.fc.weight)
            # stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
            # self.model.fc.bias.data.uniform_(-stdv, stdv)
            self.input_size = 224
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.set_param_requires_grad()
            in_features = self.model.fc.in_features 
            self.model.fc = torch.nn.Linear(in_features=in_features, out_features=self.num_classes)
            self.input_size = 224
        elif self.model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.set_param_requires_grad() #4096
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, self.num_classes)
            self.input_size = 224
        elif self.model_name == 'vgg':
            self.model = models.vgg19_bn(pretrained=True)
            self.set_param_requires_grad() # 4096
            in_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(in_features, self.num_classes)
            self.input_size = 224
        elif self.model_name == "squeezenet":
            self.model = models.squeezenet1_0(pretrained=True)
            self.set_param_requires_grad(self.model)
            self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            self.model.num_classes = self.num_classes
            self.input_size = 224
        elif self.model_name == "densenet":
            self.model = models.densenet121(pretrained=True)
            self.set_param_requires_grad(self.model)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, self.num_classes) 
            self.input_size = 224
        elif self.model_name == 'inception': #### NEED TO FIX ####
            raise NotImplementedError('inception is not yet implemented')
            self.model = models.inception_v3(pretrained=True)
            self.set_param_requires_grad() 
            num_ftrs = self.model.AuxLogits.fc.in_features # 768, Handle the auxilary net
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            num_ftrs = self.model.fc.in_features # Handle the primary net
            self.model.fc = nn.Linear(num_ftrs,self.num_classes)
            self.input_size = 299
        elif self.model_name == 'efficientnet-b3':
            from efficientnet_pytorch import EfficientNet
            self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=self.num_classes)
            self.input_size = 224
        else:
            raise ValueError(f'Expected alexnet, vgg, resnet, or inception, but received {self.model_name}..')

    def set_param_requires_grad(self):
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
