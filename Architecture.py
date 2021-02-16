import torch
import torch.nn as nn


def rollTensor(T, N):
    if (N > 0):
        k = 0
        tiles = 4
        tileShape = int(T.shape[3] / tiles)
        for i in range(tiles):
            for j in range(tiles):
                set = torch.linspace(k, k + 16 * (N - 1), N).numpy()
                T[:, set] = torch.roll(T[:, set], shifts=(tileShape * i, tileShape * j), dims=(0, 1))
                k += 1

    return T


class block(nn.Module):
    def __init__(self, in_channels, stride=1, rollSets=0):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.identity_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stride = stride
        self.rollSets = rollSets

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))

        x = rollTensor(x, N=self.rollSets)

        x = self.relu(self.bn2(self.conv2(x)))

        if self.stride != 1:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class MyNet(nn.Module):
    def __init__(self, layers=[1, 1, 1, 1], vector_coordinates=12576, rollSets=[1, 1, 1, 1]):
        super(MyNet, self).__init__()
        self.channels = 256

        self.conv1 = nn.Conv2d(1, self.channels, kernel_size=4, stride=4, padding=0)  # 1*1->256*256
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()
        self.rollSets = rollSets[0]

        # Essentially the entire MyNet architecture are in these 3 lines below
        self.layer1 = self._make_layer(layers[0], rollSets[1])  # 256->128
        self.layer2 = self._make_layer(layers[1], rollSets[2])  # 128->64
        self.layer3 = self._make_layer(layers[2], rollSets[3])  # 64->32

        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1)  # 32*256->16*256
        self.bn2 = nn.BatchNorm2d(self.channels)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.channels, self.channels)
        self.fc2 = nn.Linear(self.channels, vector_coordinates)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 256

        x = rollTensor(x, N=self.rollSets)

        x = self.layer1(x)  # 128
        x = self.layer2(x)  # 64
        x = self.layer3(x)  # 32

        x = self.relu(self.bn2(self.conv2(x)))  # 16

        x = self.avgpool(x)  # 1
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def _make_layer(self, num_residual_blocks, rollSets):
        layers = []
        layers.append(block(self.channels, stride=2, rollSets=rollSets))

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.channels, rollSets=rollSets))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    # %%timeit -n 4 -r 4
    net = MyNet()
    y = net(torch.randn(2, 1, 1024, 1024)).to("cpu")
    print(sum(p.numel() for p in net.parameters()))  # Number of trainable parameters
    print(y.size())
