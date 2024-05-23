from torch import nn
import torch


"""

What is done here is 
    - replaced the l
    - implmented the skip connection. 
    - replaced earch maxpooling with conv layer. 


The second round of changes is to 


Might need 

"""

class resVGGshallow(nn.Module):  #
    def __init__(self):
        super(resVGGshallow, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding =1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim1 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding = 0)  # Matching dimensions for skip connection

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding =1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim2 = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding = 0)  # Matching dimensions for skip connection

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding = 1),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding =1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim3 = nn.Conv2d(128, 128, kernel_size=1, stride=2, padding = 0)  # Matching dimensions for skip connection

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)  # Final output layer

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        # needed this downsampling. 
        out1 = self.match_dim1(out1)
        out2 += out1
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3 = self.match_dim2(out3)
        out4 += out3
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out5 = self.match_dim3(out5)
        out6 += out5
        
        out = self.global_avg_pool(out6)
        out = torch.flatten(out, 1)  # Flatten the tensor
        output = self.fc(out)
        return output



class resVGG(nn.Module):
    def __init__(self):
        super(resVGG, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"), nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim1 = nn.Conv2d(32, 32, kernel_size=1, stride=2, padding=0)  # Matching dimensions for skip connection

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim2 = nn.Conv2d(64, 64, kernel_size=1, stride=2, padding=0)  # Matching dimensions for skip connection

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim3 = nn.Conv2d(128, 128, kernel_size=1, stride=2, padding=0)  # Matching dimensions for skip connection

        # New fourth block
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Replacing MaxPool2d with Conv2d
        )
        self.match_dim4 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)  # Matching dimensions for skip connection

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)  # Adjusted for new block output channels

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out1 = self.match_dim1(out1)
        out2 += out1

        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3 = self.match_dim2(out3)
        out4 += out3

        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out5 = self.match_dim3(out5)
        out6 += out5

        # Forward pass for the new fourth block
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out7 = self.match_dim4(out6)  # Matching dimensions from the previous block
        out8 += out7

        out = self.global_avg_pool(out8)
        out = torch.flatten(out, 1)  # Flatten the tensor
        output = self.fc(out)
        return output