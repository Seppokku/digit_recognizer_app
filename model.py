import torch
from torch import nn


class LocalizationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 5, stride=2),
            nn.GELU(),
            nn.Conv2d(8, 4, 5),
            nn.GELU()
        )

        # classification head: n_outputs = n_classes
        self.clf_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        # box regression head: n_outputs = n_coords (4)
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, pic: torch.Tensor):
        #print(pic.size(),type(pic))
        pic = self.backbone(pic)
        #print(pic.size(), type(pic))
        clf_out = self.clf_head(pic)
        #print(clf_out.size(), type(clf_out))
        box_out = self.box_head(pic) 
        #rint(box_out.size(), type(box_out))
        return clf_out, torch.sigmoid(box_out)