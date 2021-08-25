import os
import torchvision.transforms as t

from PIL import Image


def check_image(file):
    """

    :param file:
    :return:
    """
    return any(file.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp',
                                                          '.BMP', '.tif', '.TIF', '.tiff', '.TIFF'])


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
    image = t.ToPILImage()(image)
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
