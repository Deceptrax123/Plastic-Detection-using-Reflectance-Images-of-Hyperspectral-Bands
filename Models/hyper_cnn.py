import torch
from torch.nn import Module, Conv2d, Dropout2d, MaxPool2d, ReLU, Upsample, Tanh, AdaptiveAvgPool2d, BatchNorm2d, LeakyReLU, ConvTranspose2d
from torchsummary import summary


class HyperCNN(Module):
    def __init__(self):
        super(HyperCNN, self).__init__()

        self.conv1 = Conv2d(in_channels=5, out_channels=8,
                            kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.relu1 = LeakyReLU(negative_slope=0.2)
        self.max1 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = Conv2d(in_channels=8, out_channels=16,
                            kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = BatchNorm2d(16)
        self.relu2 = LeakyReLU(negative_slope=0.2)
        self.max2 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = Conv2d(in_channels=16, out_channels=32,
                            kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = BatchNorm2d(32)
        self.relu3 = LeakyReLU(negative_slope=0.2)
        self.max3 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv4 = Conv2d(in_channels=32, out_channels=64,
                            kernel_size=(3, 3), stride=1, padding=1)
        self.bn4 = BatchNorm2d(64)
        self.relu4 = LeakyReLU(negative_slope=0.2)
        self.max4 = MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.conv5 = Conv2d(in_channels=64, out_channels=128,
                            kernel_size=(3, 3), stride=1, padding=1)
        self.bn5 = BatchNorm2d(128)
        self.relu5 = LeakyReLU(negative_slope=0.2)
        self.max5 = MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv6 = Conv2d(in_channels=128, out_channels=256,
                            kernel_size=(3, 3), stride=1, padding=1)
        self.bn6 = BatchNorm2d(256)
        self.relu6 = LeakyReLU(negative_slope=0.2)
        self.max6 = MaxPool2d(kernel_size=(1, 2), stride=2)

        self.dconv1 = ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn7 = BatchNorm2d(128)
        self.relu7 = ReLU()
        self.up1 = Upsample(scale_factor=2)

        self.dconv2 = ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn8 = BatchNorm2d(64)
        self.relu8 = ReLU()
        self.up2 = Upsample(scale_factor=2)

        self.dconv3 = ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn9 = BatchNorm2d(32)
        self.relu9 = ReLU()
        self.up3 = Upsample(scale_factor=2)

        self.dconv4 = ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.bn10 = BatchNorm2d(16)
        self.relu10 = ReLU()
        self.up4 = Upsample(scale_factor=2)

        self.dconv5 = ConvTranspose2d(
            in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1)
        self.bn11 = BatchNorm2d(8)
        self.relu11 = ReLU()
        self.up5 = Upsample(scale_factor=2)

        self.dconv6 = ConvTranspose2d(
            in_channels=8, out_channels=5, kernel_size=(3, 3), stride=1, padding=1)
        self.bn12 = BatchNorm2d(5)
        self.relu12 = ReLU()
        self.up6 = Upsample(scale_factor=2)

        self.adaptive = AdaptiveAvgPool2d(output_size=(1300, 1600))
        self.reflectance = Conv2d(
            in_channels=5, out_channels=5, kernel_size=(1, 1))
        self.tanh = Tanh()

        self.dp1 = Dropout2d(0.8)
        self.dp2 = Dropout2d(0.8)
        self.dp3 = Dropout2d(0.8)
        self.dp4 = Dropout2d(0.8)
        self.dp5 = Dropout2d(0.8)
        self.dp6 = Dropout2d(0.8)
        self.dp7 = Dropout2d(0.8)
        self.dp8 = Dropout2d(0.8)
        self.dp9 = Dropout2d(0.8)
        self.dp10 = Dropout2d(0.8)
        self.dp11 = Dropout2d(0.8)
        self.dp12 = Dropout2d(0.8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max1(x)
        x = self.dp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.max2(x)
        x = self.dp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.max3(x)
        x = self.dp3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.max4(x)
        x = self.dp4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.max5(x)
        x = self.dp5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.max6(x)
        x = self.dp6(x)

        x = self.dconv1(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.up1(x)
        x = self.dp7(x)

        x = self.dconv2(x)
        x = self.bn8(x)
        x = self.relu8(x)
        x = self.up2(x)
        x = self.dp8(x)

        x = self.dconv3(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.up3(x)
        x = self.dp9(x)

        x = self.dconv4(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.up4(x)
        x = self.dp10(x)

        x = self.dconv5(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.up5(x)
        x = self.dp11(x)

        x = self.dconv6(x)
        x = self.bn12(x)
        x = self.tanh(x)
        x = self.up6(x)
        x = self.dp12(x)

        x = self.adaptive(x)
        x = self.tanh(x)

        return x


# if __name__ == '__main__':
#     model = HyperCNN()
#     summary(model, input_size=(5, 1300, 1600), device='cpu')
