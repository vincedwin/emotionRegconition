import torch
import torch.nn as nn

class Deep_Emotion(nn.Module):
    """Initializing"""
    def __init__(self):
        super(Deep_Emotion,self).__init__()

        # INc, OUTc, kernel size
        self.conv1 = nn.Conv2d(1,10,3)

        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    ## attention layer
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nn.functional.affine_grid(theta, x.size())
        x = nn.functional.grid_sample(x, grid)
        return x

    def forward(self,input):
        ''' implementation '''
        out = self.stn(input) ## ATTENTION UNIT

        out = nn.functional.relu(self.conv1(out))
        out = self.conv2(out)
        out = nn.functional.relu(self.pool2(out))

        out = nn.functional.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = nn.functional.relu(self.pool4(out))

        out = nn.functional.dropout(out)
        out = out.view(-1, 810)
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)

        return out