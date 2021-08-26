import torch.nn as nn


class Discriminator(nn.Module):
    """

    """

    def __init__(self, image_size):
        """

        :param image_size:
        """

        super(Discriminator, self).__init__()

        feature_size = image_size // 32

        # first conv layer
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=(1, 1)),
            nn.LeakyReLU()
        )

        # second conv layer
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU()
        )

        # discriminator block channels
        in_channels = [64, 128, 128, 256, 256, 512, 512, 512]
        out_channels = [128, 128, 256, 256, 512, 512, 512, 512]

        # generator for discriminator blocks
        gen_block = (lambda in_c, out_c, s: nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=1, stride=s, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ))

        self.discriminator_blocks = nn.Sequential(*[gen_block(in_c, out_c, (i % 2 + 1, i % 2 + 1))
                                                    for i, (in_c, out_c) in enumerate(zip(in_channels, out_channels))])

        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_size * feature_size, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        """

        :param x:
        :return:
        """

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.discriminator_blocks(out)
        batch, *size = out.shape

        return self.classifier(out.view(batch, -1))
