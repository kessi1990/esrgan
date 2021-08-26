import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset

import os
import multiprocessing as mp

from models.generator import Generator
from models.discriminator import Discriminator
from models.content_loss import ContentLoss
from datasets.train_dataset import TrainDataset
from training import train
from utils import utils


if __name__ == '__main__':
    mp.freeze_support()

    # hyperparameters
    learning_rate = 0.0001
    batch_size = 16
    epochs = 10000
    alpha = 0.01
    eta = 1
    lambda_ = 0.005
    hyperparameters = {'batch_size': batch_size, 'alpha': alpha, 'eta': eta, 'lambda': lambda_}

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
    test_img = utils.test_img(os.path.join('data', 'test_img.jpg'))
    train_dataset = TrainDataset(path_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=True)

    losses = []

    for e in range(1, epochs + 1):
        # train GAN
        generator_loss, discriminator_loss = train(generator, discriminator, train_dataloader,
                                                   generator_optimizer, discriminator_optimizer,
                                                   perceptual_criterion, content_criterion, adversarial_criterion,
                                                   hyperparameters, epoch=e)

        # save total losses from both generator and discriminator for plotting
        losses.append((generator_loss, discriminator_loss))

        if e % 100 == 0:
            # save model
            # TODO: only save current model if current model is more accurate than last stored (best) model
            torch.save(generator.state_dict(), os.path.join('output', f'generator_{e}.pth'))
            torch.save(discriminator.state_dict(), os.path.join('output', f'discriminator_{e}.pth'))

            # generate and save super resolution image from test image for each epoch
            if test_img is not None:
                with torch.no_grad():
                    super_resolution = generator(test_img.unsqueeze(dim=0).to(device))
                    utils.save_image(super_resolution.detach(), os.path.join('output', f'super_resolution_{e}.png'))
