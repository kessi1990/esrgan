import torch
import time
import logging


logger = logging.getLogger(__name__)


def calc_time(func):
    """
    decorator function for train(), measures and logs time for each training epoch
    :param func:
    :return:
    """

    def inner_func(*args, **kwargs):
        start = time.time()
        losses = func(*args, **kwargs)
        end = time.time()
        logging.info(f'epoch {kwargs["epoch"]}: time: {end - start:.2f}, avg generator loss {losses[0]:.4f}, '
                     f'avg discriminator loss {losses[1]:.4f}')

        return losses

    return inner_func


@calc_time
def train(generator, discriminator, train_dataloader, generator_optimizer, discriminator_optimizer,
          perceptual_criterion, content_criterion, adversarial_criterion, params, **kwargs):
    """

    :param generator:
    :param discriminator:
    :param train_dataloader:
    :param generator_optimizer:
    :param discriminator_optimizer:
    :param perceptual_criterion:
    :param content_criterion:
    :param adversarial_criterion:
    :param params:
    :param kwargs:
    :return:
    """

    generator_losses = []
    discriminator_losses = []

    for i, (low_res, high_res) in enumerate(train_dataloader):
        #####################################################################################
        # 1. discriminator network:                                                         #
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
        label_real = torch.full((params['batch_size'], 1), 1, dtype=low_res.dtype)
        label_fake = torch.full((params['batch_size'], 1), 0, dtype=low_res.dtype)

        # compute adversarial loss for real (high resolution) and fake (super resolution) images
        loss_real_d = adversarial_criterion(output_real - torch.mean(output_fake), label_real)
        loss_fake_d = adversarial_criterion(output_fake - torch.mean(output_real), label_fake)

        # compute total discriminator loss
        total_loss_d = loss_real_d + loss_fake_d

        # loss backwards
        total_loss_d.backward()
        discriminator_optimizer.step()

        #####################################################################################
        # 2. generator network:                                                             #
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
        total_loss_g = params['alpha'] * perceptual_loss + params['lambda'] * adversarial_loss + params['eta'] * content_loss

        # loss backwards
        total_loss_g.backward()
        generator_optimizer.step()

        # zero gradients
        generator.zero_grad()

        generator_losses.append(total_loss_g.item())
        discriminator_losses.append((total_loss_d.item()))

    return sum(generator_losses) / len(generator_losses), sum(discriminator_losses) / len(discriminator_losses)
