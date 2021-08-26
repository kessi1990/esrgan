import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as t

from PIL import Image


def check_image(file):
    """

    :param file:
    :return:
    """

    return any(file.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp',
                                                          '.BMP', '.tif', '.TIF', '.tiff', '.TIFF'])


def test_img(image_path):
    """

    :param image_path:
    :return:
    """

    if check_image(image_path):
        return t.ToTensor()(Image.open(image_path).convert('RGB'))
    else:
        return None


def load_images(directory):
    """

    :param directory:
    :return:
    """

    return [os.path.join(directory, image) for image in os.listdir(directory) if check_image(image)]


def save_image(image, path):
    """

    :param image:
    :param path:
    :return:
    """

    image = t.ToPILImage()(image.squeeze())
    image.save(path)


def transform_low(low_res_size):
    """

    :param low_res_size:
    :return:
    """

    return t.Compose([
        t.ToPILImage(),
        t.Resize((low_res_size, low_res_size), interpolation=Image.BICUBIC),
        t.ToTensor()
    ])


def transform_high(image_size):
    """

    :param image_size:
    :return:
    """

    return t.Compose([
        t.RandomCrop((image_size, image_size)),
        t.RandomVerticalFlip(),
        t.RandomHorizontalFlip(),
        t.RandomRotation(degrees=90),
        t.ToTensor()
    ])


def normalize():
    """

    :return:
    """

    return t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)


def plot_loss(loss):
    """

    :param loss:
    :return:
    """

    loss = np.array(loss)
    generator_loss = loss[:, 0]
    discriminator_loss = loss[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    fig.suptitle('avg loss per epoch')

    axes[0].plot(generator_loss)
    axes[0].set(xlabel='epoch', ylabel='generator loss')

    axes[1].plot(discriminator_loss)
    axes[1].set(xlabel='epoch', ylabel='discriminator loss')

    plt.savefig(os.path.join('output', 'loss.png'))
    plt.close()
