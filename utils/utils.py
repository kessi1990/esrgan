import os
import matplotlib.pyplot as plt
import torchvision.transforms as t

from PIL import Image


def check_image(file):
    """
    function to check whether the passed file is an image or not, which is done by checking the file's extension.
    supported image types are jpeg, png, bmp, tiff.
    :param file: file of arbitrary type
    :return: boolean
    """

    return any(file.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp',
                                                          '.BMP', '.tif', '.TIF', '.tiff', '.TIFF'])


def test_img(image_path):
    """
    loads the test image, converts its color space and returns it as pytorch tensor
    :param image_path: path to the file
    :return: pytorch tensor with image data or None
    """

    if check_image(image_path):
        return t.ToTensor()(Image.open(image_path).convert('RGB'))
    else:
        return None


def load_images(directory):
    """
    loads all paths of image files in a directory based on the passed directory path
    :param directory: path to the image directory
    :return: a list of all paths of images files contained in the given directory
    """

    return [os.path.join(directory, image) for image in os.listdir(directory) if check_image(image)]


def save_image(image, path):
    """
    saves image (data passed as pytorch tensor) to a directory
    :param image: pytorch tensor containing the image data
    :param path: path the the directory
    :return: None
    """

    image = t.ToPILImage()(image.squeeze())
    image.save(path)


def transform_low(low_res_size):
    """
    constructs a transformation pipeline for image data. this pipeline expects a pytorch tensor, resizes the image data
    to the passed size by applying bicubic interpolation and returns the downscaled image as pytorch tensor
    :param low_res_size: integer value defining the size of the low resolution
    :return: a transformation pipeline
    """

    return t.Compose([
        t.ToPILImage(),
        t.Resize((low_res_size, low_res_size), interpolation=Image.BICUBIC),
        t.ToTensor()
    ])


def transform_high(image_size):
    """
    constructs a transformation pipeline for image data. this pipeline expects a PIL image, crops a quadratic chunk
    (fixed in size) at random position out of the original image, applies random vertical and horizontal flips and
    rotation and returns the the image data as pytorch tensor
    :param image_size: integer value defining the size of the cropped out chunks
    :return: a transformation pipeline
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
    normalizes the image data
    :return: a function which is applied to image data
    """

    return t.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)


def plot_loss(generator_loss, discriminator_loss):
    """
    small function to plot the gathered generator and discriminator loss during training
    :param generator_loss: numpy array containing the generator loss during training
    :param discriminator_loss: numpy array containing the discriminator loss during training
    :return: None
    """

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    fig.title('avg loss per epoch')

    axs[0].plot(generator_loss)
    axs[0].set(xlabel='epoch', ylabel='generator loss')

    axs[1].plot(discriminator_loss)
    axs[1].set(xlabel='epoch', ylabel='discriminator loss')

    plt.savefig(os.path.join('output', 'loss.png'))
    plt.close()
