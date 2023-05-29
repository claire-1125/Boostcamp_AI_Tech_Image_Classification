##################################### baseline code #####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


# smoothing값은 hyperparameter
# https://jeonghwarr.github.io/tips/label_smoothing/
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.2, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))  # -true_dist * pred : cross entropy loss


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        # return 1 - f1.mean()
        return f1.mean()



##### Arcface Loss
# https://aimaster.tistory.com/93
class ArcMarginProduct(nn.Module):
    # in_feature : model의 output size (아니면 batch size...??) 아니면 224*224
    # embedding_size: The size of the embeddings that you pass into the loss function. 
    # For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, 
    # then set embedding_size to 512.
    def __init__(self, in_feature=64, classes=18, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.classes = classes
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(classes, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        """
        input shape (N, in_features)
        """

        # cos(theta)
        # input, self.weight를 normalize해줌으로써 길이 1인 구 위에 위치
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))  # classes 크기로 나옴

        # cos(theta + m)
        # cos(theta+m) = cos(theta) * cos(m) + sin(theta) * sin(m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # (1-cos^2)^{1/2}=sin
        phi = cosine * self.cos_m - sine * self.sin_m 

        if self.easy_margin:  # easy margin? theta + m > phi 인 상황을 고려하지 않기 위함.
            phi = torch.where(cosine > 0, phi, cosine)  # condition, condition의 true일 때 값, condition이 false일 때 값
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'arcface' : ArcMarginProduct
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)

        # create_fn이 nn.CrossEntropyLoss일 경우
        # 이 생성자의 argument로 weight 리스트를 넣어주면
        # weighted cross entropy를 사용하게 된다.
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
