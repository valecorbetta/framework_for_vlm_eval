import timm
import torch.nn as nn
import torch
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        if isinstance(self.p, int):
            # If self.p is an integer
            return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p)
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
            )
        else:
            # If self.p is a PyTorch tensor
            return (
                self.__class__.__name__
                + "("
                + "p="
                + "{:.4f}".format(self.p.data.tolist()[0])
                + ", "
                + "eps="
                + str(self.eps)
                + ")"
            )


class EfficientNetMammo(nn.Module):
    def __init__(
        self,
        name: str = "tf_efficientnet_b2_ns-detect",
        pretrained=False,
        in_chans=1,
        p=3,
        p_trainable=False,
        eps=1e-6,
        get_features=False,
    ):
        super().__init__()
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)
        clsf = model.default_cfg["classifier"]
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()
        self.out_dim = n_features

        self.fc = nn.Linear(n_features, 1)
        self.model = model
        self.get_features = get_features
        self.pool = nn.Sequential(
            GeM(p=p, eps=eps, p_trainable=p_trainable), nn.Flatten()
        )

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.pool(x)
        return x
