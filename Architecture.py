import torch
import torch.nn as nn


class rollTensor(nn.Module):
    def __init__(self, channels, size, rollSets=0, tiles=4):
        super(rollTensor, self).__init__()
        self.LeftRolledIdentity, self.RightRolledIdentity = self._rollTensorfun(channels, size, rollSets, tiles)

    def forward(self, x):
        # x = torch.einsum('inj,mijk,ikl->minl', self.LeftRolledIdentity, x,
        #                  self.RightRolledIdentity)
        x = torch.einsum('mijk,ikl->mijl', x, self.RightRolledIdentity)
        x = torch.einsum('inj,mijl->minl', self.LeftRolledIdentity, x)
        return x

    def _rollTensorfun(self, channels, size, rollSets, tiles):
        LeftI = torch.eye(size).unsqueeze(0)
        LeftI = torch.repeat_interleave(LeftI, channels, dim=0)
        RightI = LeftI.clone()
        I = torch.eye(size)

        k = 0
        tileShape = int(size / tiles)
        for N in range(rollSets):
            for i in range(tiles):
                for j in range(tiles):
                    LeftI[k] = torch.roll(I, (tileShape * i), 0)
                    RightI[k] = torch.roll(I, (tileShape * j), 1)
                    k += 1
        return LeftI, RightI


class block(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.identity_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stride = stride


    def forward(self, x):

        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
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
        self.layer1 = self._make_layer(layers[0])  # 256->128
        self.layer2 = self._make_layer(layers[1])  # 128->64
        self.layer3 = self._make_layer(layers[2])  # 64->32

        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1)  # 32*256->16*256
        self.bn2 = nn.BatchNorm2d(self.channels)

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.channels, self.channels)
        self.fc2 = nn.Linear(self.channels, vector_coordinates)

        self.rollTensor1 = rollTensor(self.channels, size=256, rollSets=rollSets[0])
        self.rollTensor2 = rollTensor(self.channels, size=128, rollSets=rollSets[1])
        self.rollTensor3 = rollTensor(self.channels, size=64, rollSets=rollSets[2])
        self.rollTensor4 = rollTensor(self.channels, size=32, rollSets=rollSets[3])

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))  # 256

        # x = rollTensor(x, N=self.rollSets)
        x = self.rollTensor1(x)

        x = self.layer1(x)  # 128
        x = self.rollTensor2(x)
        x = self.layer2(x)  # 64
        x = self.rollTensor3(x)
        x = self.layer3(x)  # 32
        x = self.rollTensor4(x)

        x = self.relu(self.bn2(self.conv2(x)))  # 16

        x = self.avgpool(x)  # 1
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def _make_layer(self, num_residual_blocks):
        layers = []
        layers.append(block(self.channels, stride=2))

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.channels))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    # %%timeit -n 4 -r 4

    # channels=16
    # size=4
    # test = rollTensor(channels=channels, size=size, rollSets=1, tiles=2)
    # x=torch.arange(16).reshape(4,4).unsqueeze(0).unsqueeze(0)
    #
    # x=torch.repeat_interleave(x, channels, dim=1).float()
    # x = torch.repeat_interleave(x, 2, dim=0).float()
    #
    # y = test(x.to("cpu"))
    # print(y.size())
    # x = x.numpy()
    # y = y.numpy()

    net = MyNet()
    y = net(torch.randn(2, 1, 1024, 1024)).to("cpu")
    print(sum(p.numel() for p in net.parameters()))  # Number of trainable parameters
    print(y.size())

