from torch import nn
from torchvision.models import ResNet101_Weights, resnet101

class PhotoRotateModel(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = nn.Sequential(*list(resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).children())[:-1])
        for param in resnet.parameters():
                param.requires_grad = False
        resnet.eval()

        self.model = nn.Sequential(
             resnet,
             nn.Flatten(start_dim=1),
             nn.Tanh(),
             nn.LazyLinear(1),
        )


    def forward(self, data):
        return self.model(data)
