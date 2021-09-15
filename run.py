import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset

import os
import logging
import numpy as np
import multiprocessing as mp

from models.generator import Generator
from models.discriminator import Discriminator
from models.content_loss import ContentLoss
from datasets.train_dataset import TrainDataset
from training import train
from utils import utils

# config for logger
logging.basicConfig(filename=os.path.join('output', 'run.log'), filemode='a', format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


if __name__ == '__main__':
    mp.freeze_support()
    logger = logging.getLogger(__name__)

    # hyperparameters
    learning_rate = 0.0001
    batch_size = 16
    epochs = 1000
    alpha = 0.01
    eta = 1
    lambda_ = 0.005
    hyperparameters = {'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs,
                       'alpha': alpha, 'eta': eta, 'lambda': lambda_}

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log info about device and hyperparameters
    logger.info(f'device: {device}')
    logger.info(f'hyperparameters: {hyperparameters}')

    # create model
    generator = Generator(nr_blocks=23).to(device)
    discriminator = Discriminator(scale=4).to(device)

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
    test_img = utils.test_img(os.path.join('data', 'test_img.jpg')).unsqueeze(dim=0).to(device)
    train_dataset = TrainDataset(path_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=True)

    gen_loss = np.zeros(epochs)
    disc_loss = np.zeros(epochs)

    for e in range(epochs):
        # train GAN
        generator_loss, discriminator_loss = train(generator, discriminator, train_dataloader,
                                                   generator_optimizer, discriminator_optimizer,
                                                   perceptual_criterion, content_criterion, adversarial_criterion,
                                                   hyperparameters, device, epoch=e)

        # save total loss from both generator and discriminator for future plotting
        gen_loss[e] = generator_loss
        disc_loss[e] = discriminator_loss

        if e % 10 == 0 or e == epochs:
            # save model
            # TODO: only save current model if current model is more accurate than last stored (best) model
            torch.save(generator.state_dict(), os.path.join('output', f'generator_{e}.pth'))
            torch.save(discriminator.state_dict(), os.path.join('output', f'discriminator_{e}.pth'))
            torch.save(generator_optimizer.state_dict(), os.path.join('output', f'gen_optimizer_{e}.pth'))
            torch.save(discriminator_optimizer.state_dict(), os.path.join('output', f'disc_optimizer_{e}.pth'))
            logger.info(f'model saved at epoch {e}')

            # generate and save super resolution image from test image
            if test_img is not None:
                with torch.no_grad():
                    super_resolution = generator(test_img)
                    utils.save_image(super_resolution.detach(), os.path.join('output', f'super_resolution_{e}.png'))
                    logger.info(f'generated super resolution image at epoch {e}')

    # plot avg loss for each epoch
    utils.plot_loss(gen_loss, disc_loss)
    logger.info('training complete!')
