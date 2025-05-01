import torch.nn as nn, torchvision.models as models

class ResNetTwoHead(nn.Module):
    def __init__(self, n_plants, n_diseases, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(
            weights='IMAGENET1K_V2' if pretrained else None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.plant_head   = nn.Linear(in_feats, n_plants)
        self.disease_head = nn.Linear(in_feats, n_diseases)

    def forward(self, x):
        feats = self.backbone(x)
        return self.plant_head(feats), self.disease_head(feats)