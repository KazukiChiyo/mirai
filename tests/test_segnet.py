import torch
from mirai.backend import UNet
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=16, feature_scale=2).to(device)
summary(model, input_size=(3, 360, 480), batch_size=4)
X = torch.randn(1, 3, 360, 480).to(device)
output = model(X)
assert(output.shape == torch.Size([1, 16, 360, 480]))
