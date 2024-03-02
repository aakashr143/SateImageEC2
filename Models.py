import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def weights_init(m):
    torch.nn.init.xavier_normal_(m.weight)
    torch.nn.init.constant_(m.bias, 0)
    return m


def create_classifier_layer_no_features(in_features, num_classes):
    layer = nn.Sequential(
        nn.Dropout(0.2),
        weights_init(nn.Linear(in_features, 1024)),
        nn.ReLU(),
        weights_init(nn.Linear(1024, num_classes)),
        nn.LogSoftmax(dim=1)
    )
    return layer


class LandUseModelVisionTransformerB16NoFeatures(nn.Module):
    def __init__(self, num_classes, device="cpu"):
        super(LandUseModelVisionTransformerB16NoFeatures, self).__init__()
        self.name = "visiontransformerb16_nofeatures"
        self.device = device

        self.base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        for params in self.base_model.parameters():
            params.requires_grad = False

        self.classifier = create_classifier_layer_no_features(768, num_classes)

    def forward(self, x):
        out = self.base_model._process_input(x)

        n = out.shape[0]

        batch_class_token = self.base_model.class_token.expand(n, -1, -1)
        out = torch.cat([batch_class_token, out], dim=1)

        out = self.base_model.encoder(out)

        out = out[:, 0]

        out = out.reshape(out.shape[0], -1)

        return self.classifier(out)
