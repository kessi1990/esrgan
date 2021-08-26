import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset

import os
import time
import multiprocessing as mp

from torchvision.transforms import transforms as t
from PIL import Image

from models.generator import Generator
from models.discriminator import Discriminator
from models.content_loss import ContentLoss
from datasets.train_dataset import TrainDataset
from utils import utils_img


def train():

    for i, (low_res, high_res) in enumerate(train_dataloader):
        #####################################################################################
        # 1. discriminator network:
        #####################################################################################

        # zero gradients
        discriminator_optimizer.zero_grad()

        # generate super resolution from low resolution
        super_res = generator(low_res)

        # discriminate real (high resolution = groundtruth) and fake (generated super resolution)
        # --> discriminator learns to differentiate between real and fake data
        output_real = discriminator(high_res)
        output_fake = discriminator(super_res.detach())

        # generate labels for real (high resolution = groundtruth) samples = 1 and fake (super resolution) samples = 0
        label_real = torch.full((batch_size, 1), 1, dtype=low_res.dtype)
        label_fake = torch.full((batch_size, 1), 0, dtype=low_res.dtype)

        # compute adversarial loss for real (high resolution) and fake (super resolution) images
        loss_real_d = adversarial_criterion(output_real - torch.mean(output_fake), label_real)
        loss_fake_d = adversarial_criterion(output_fake - torch.mean(output_real), label_fake)

        # compute total discriminator loss
        total_loss_d = loss_real_d + loss_fake_d

        # loss backwards
        total_loss_d.backward()
        discriminator_optimizer.step()

        #####################################################################################
        # 2. generator network:
        #####################################################################################

        # zero gradients
        generator.zero_grad()

        # generate super resolution from high_res
        super_res = generator(low_res)

        # discriminate real (high resolution = groundtruth) and fake (generated super resolution)
        # --> generator learns to fool discriminator
        # --> discriminator can no longer differentiate between real and fake data
        output_real = discriminator(high_res.detach())
        output_fake = discriminator(super_res)

        # compute perceptual loss = mean absolute error (L1) of pixels
        perceptual_loss = perceptual_criterion(super_res, high_res.detach())

        # compute content loss (L1) using pre-trained vgg19 as feature extractor
        content_loss = content_criterion(super_res, high_res.detach())

        # compute adversarial loss
        adversarial_loss = adversarial_criterion(output_fake - torch.mean(output_real), label_real)

        # compute total generator loss
        total_loss_g = alpha * perceptual_loss + lambda_ * adversarial_loss + eta * content_loss

        # loss backwards
        total_loss_g.backward()
        generator_optimizer.step()

        # zero gradients
        generator.zero_grad()


if __name__ == '__main__':
    mp.freeze_support()

    # hyperparameters
    learning_rate = 0.0001
    batch_size = 16
    epochs = 10000
    alpha = 0.01
    eta = 1
    lambda_ = 0.005

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
    generator = Generator(nr_blocks=23).to(device)
    discriminator = Discriminator(image_size=128).to(device)

    # loss = 0.01 * perceptual loss + content loss + 0.005 * adversarial loss
    perceptual_criterion = nn.L1Loss()
    content_criterion = ContentLoss()
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # switch to train mode
    generator.train()
    discriminator.train()

    # datasets, dataloaders and test image
    path_train = os.path.join('data', 'train_data')
    # test_img = t.Compose([t.ToTensor()])(Image.open('data/test_img.jpg').convert('RGB'))
    train_dataset = TrainDataset(path_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=True)

    for e in range(epochs):
        # train GAN
        start = time.time()
        train()
        end = time.time()
        print(f'epoch {e}, time: {end - start}')

        if e % 100 == 0:
            # save model
            # TODO: only save current model if current model is more accurate than last stored (best) model
            torch.save(generator.state_dict(), os.path.join('output', f'generator_{e}.pth'))
            torch.save(discriminator.state_dict(), os.path.join('output', f'discriminator_{e}.pth'))

            # generate and save super resolution image from test image for each epoch
            with torch.no_grad():
                start = time.time()
                super_resolution = torch.randn((3, 128, 128))  # generator(test_img.unsqueeze(dim=0))
                end = time.time()
                utils_img.save_image(super_resolution.detach(), os.path.join('output', f'super_resolution_{e}.png'))
