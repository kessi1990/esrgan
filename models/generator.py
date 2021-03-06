import torch
import torch.nn as nn
import torch.nn.functional as functional


class ResidualBlock(nn.Module):
    """
    residual dense block
    blocks 1 - 4: conv2d + leaky relu
    block 5: conv2d
    """

    def __init__(self, channels, growth_channels, scale):
        """
        init / constructor method
        :param channels: integer, defines number of in channels
        :param growth_channels: integer, defines number of growth channels
        :param scale: integer, defines scaling factor
        """

        super(ResidualBlock, self).__init__()
        self.scale = scale

        # generator for blocks 1-4
        gen_block = (lambda i: nn.Sequential(
            nn.Conv2d(in_channels=channels + i * growth_channels, out_channels=growth_channels,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ))

        # build blocks 1 - 4
        self.conv_1 = gen_block(0)
        self.conv_2 = gen_block(1)
        self.conv_3 = gen_block(2)
        self.conv_4 = gen_block(3)

        # last conv layer without leaky relu
        self.conv_5 = nn.Conv2d(in_channels=channels + 4 * growth_channels, out_channels=channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        """
        forwards image input through residual architecture
        :param x: pytorch tensor, contains image data
        :return: pytorch tensor with image data after upscaling
        """

        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(torch.cat((x, conv_1), dim=1))
        conv_3 = self.conv_3(torch.cat((x, conv_1, conv_2), dim=1))
        conv_4 = self.conv_4(torch.cat((x, conv_1, conv_2, conv_3), dim=1))
        conv_5 = self.conv_5(torch.cat((x, conv_1, conv_2, conv_3, conv_4), dim=1))

        return torch.add(conv_5 * self.scale, x)


class ResidualInResidualDenseBlock(nn.Module):
    """
    residual in residual dense block, consists of 3 residual dense blocks
    """

    def __init__(self, channels, growth_channels, scale):
        """
        init / constructor
        :param channels: integer, defines number of in channels
        :param growth_channels: integer, defines number of growth channels
        :param scale: integer, defines scaling factor
        """

        super(ResidualInResidualDenseBlock, self).__init__()
        self.scale = scale
        self.dense_block_1 = ResidualBlock(channels, growth_channels, scale)
        self.dense_block_2 = ResidualBlock(channels, growth_channels, scale)
        self.dense_block_3 = ResidualBlock(channels, growth_channels, scale)

    def forward(self, x):
        """
        forwards image input through residual architecture
        :param x: pytorch tensor, contains image data
        :return: pytorch tensor with image data after upscaling
        """
        out = self.dense_block_1(x)
        out = self.dense_block_2(out)
        out = self.dense_block_3(out)
        return torch.add(out * self.scale, x)


class Generator(nn.Module):
    """
    generator network
    """
    def __init__(self, nr_blocks):
        """
        init / constructor
        :param nr_blocks: integer, defines number of residual in residual dense blocks
        """

        super(Generator, self).__init__()

        # number of ResidualInResidualDenseBlocks (RRDBs)
        self.nr_blocks = nr_blocks

        # input / first layer
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # network trunk containing RRDBs
        self.trunk = nn.Sequential(*[ResidualInResidualDenseBlock(channels=64, growth_channels=32, scale=0.2)
                                     for _ in range(self.nr_blocks)])

        # conv layer after RRDBs
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # upsampling layers
        self.up_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # follow up layer after upsampling layers
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # output layer
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        """
        forwards image input through generator network
        :param x: pytorch tensor, contains image data
        :return: pytorch tensor with image data after upscaling
        """

        out_1 = self.conv_1(x)
        trunk = self.trunk(out_1)

        out_2 = self.conv_2(trunk)
        out = torch.add(out_1, out_2)

        out = self.up_1(functional.interpolate(out, scale_factor=2, mode='nearest'))
        out = self.up_2(functional.interpolate(out, scale_factor=2, mode='nearest'))
        out = self.conv_3(out)

        return self.conv_4(out)
