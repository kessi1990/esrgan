import torch.nn as nn
import torch.nn.functional as functional

from torchvision.models import vgg19


class ContentLoss(nn.Module):
    """

    """
    def __init__(self):
        """

        """

        super(ContentLoss, self).__init__()

        # 36th layer of vgg19 acts as feature extractor
        self.feature_extractor = nn.Sequential(*list(vgg19(pretrained=True).features.children())[:35]).eval()

        # freeze model parameters
        for name, parameters in self.feature_extractor.named_parameters():
            parameters.requires_grad = False

    def forward(self, source, target):
        """

        :param source:
        :param target:
        :return:
        """

        return functional.l1_loss(self.feature_extractor(source), self.feature_extractor(target))
