import torch
import torchvision

class ResNet101(torch.nn.Module):
    def __init__(self, out_size=4):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        self.resnet101.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(self.resnet101.fc.in_features, out_size)
        )
    def forward(self, x):
        x = self.resnet101(x)
        return x
