import torch.nn as nn
import torch.nn.functional as functional

from torchvision.models import vgg19


class ContentLoss(nn.Module):
    """
    class for constructing and obtaining the content loss based on the layers of a pre-trained vgg19 network
    (feature extractor) as suggested in the original paper
    """
    def __init__(self):
        """
        init / constructor
        """

        super(ContentLoss, self).__init__()

        # layers of vgg19 act as feature extractor
        self.feature_extractor = nn.Sequential(*list(vgg19(pretrained=True).features.children())[:35]).eval()

        # freeze model parameters
        for name, parameters in self.feature_extractor.named_parameters():
            parameters.requires_grad = False

    def forward(self, source, target):
        """
        computes the content l1 loss based on the extracted features of both the source and target image
        :param source: pytorch tensor, contains source image data
        :param target: pytorch tensor, contains target image data
        :return: content loss
        """

        return functional.l1_loss(self.feature_extractor(source), self.feature_extractor(target))
