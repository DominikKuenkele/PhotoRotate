from torch import nn
from torchvision.models import ResNet101_Weights, resnet101


class ResnetFeatureExtractor(nn.Module):
    def __init__(
        self, pretrained=True, fine_tune=False, number_blocks=4, avgpool=True, fc=True
    ) -> None:
        super().__init__()

        self.pretrained = pretrained
        self.fine_tune = fine_tune
        self.number_blocks = number_blocks
        self.avgpool = avgpool
        self.fc = fc

        if pretrained:
            resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet101()

        if number_blocks > 4 or number_blocks < 1:
            raise AttributeError("number of blocks need to be between 1 and 3")

        self.resnet = nn.Sequential(*list(resnet.children())[:5])
        self._feature_shape = (256, 56, 56)

        if number_blocks > 1:
            self.resnet.append(resnet.layer2)
            self._feature_shape = (512, 28, 28)

        if number_blocks > 2:
            self.resnet.append(resnet.layer3)
            self._feature_shape = (1024, 14, 14)

        if number_blocks > 3:
            self.resnet.append(resnet.layer4)
            self._feature_shape = (2048, 7, 7)

        if avgpool:
            self.resnet.append(resnet.avgpool)
            self._feature_shape = (2048, 1, 1)

        if fc:
            self.resnet.append(nn.Flatten())
            self.resnet.append(resnet.fc)
            self._feature_shape = (1000,)

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()

    @property
    def feature_shape(self):
        return self._feature_shape

    def forward(self, data):
        return self.resnet(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pretrained=}, {self.fine_tune=}, {self.number_blocks=}, {self.avgpool=:=}, {self.fc=})"


class PhotoRotateModel(nn.Module):
    def __init__(self, resnet: ResnetFeatureExtractor, dropout: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            resnet,
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LazyLinear(4),
            nn.Softmax(dim=1),
        )

    def forward(self, data):
        return self.model(data)


class AttentionModule(nn.Module):
    def __init__(self, attention_dim):
        super().__init__()

        self.attention_weights = nn.Sequential(
            nn.LazyLinear(attention_dim), nn.Tanh(), nn.LazyLinear(1), nn.Softmax(dim=1)
        )

    def forward(self, features):
        attention_weights = self.attention_weights(features)
        attended_features = features * attention_weights

        return attended_features.sum(dim=1)


class PhotoRotateAttentionModel(nn.Module):
    def __init__(
        self,
        resnet: ResnetFeatureExtractor,
        dropout: int,
        *args,
        attention_dims: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            resnet,
            AttentionModule(attention_dim=attention_dims),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LazyLinear(4),
            nn.Softmax(dim=1),
        )

    def forward(self, data):
        return self.model(data)
