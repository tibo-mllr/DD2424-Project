from torch import nn


class BaseLineModel(nn.Module):  #
    def __init__(self):
        super(BaseLineModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(2048, 128), nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128, 10)
            # removed a relu down here. May or may not be necessary.
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # It seems that He = kaiming
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # Or we can also use
                # nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.fc(out)
        output = self.fc2(out)

        return output
