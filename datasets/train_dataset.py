from torch.utils.data import dataset
from PIL import Image

from utils import utils


class TrainDataset(dataset.Dataset):
    """
    dataset class, contains training data, returns already transformed data samples
    """
    def __init__(self, root, image_size=128, upscale_factor=4):
        """
        init / constructor
        :param root: String, root directory
        :param image_size: integer, defines size of image data
        :param upscale_factor: integer, defines scaling factor
        """

        super(TrainDataset, self).__init__()

        low_res_size = image_size // upscale_factor

        self.files = utils.load_images(root)
        self.transform_low = utils.transform_low(low_res_size)
        self.transform_high = utils.transform_high(image_size)
        self.normalize = utils.normalize()

    def __getitem__(self, index):
        """
        fetches data sample by key (sampled by index), applies transformation pipelines (downscaling) and normalizes
        the image data
        :param index: integer for accessing data structure
        :return: pytorch tensors containing image data of low and high resolution
        """

        file = Image.open(self.files[index]).convert('RGB')

        high_res = self.transform_high(file)
        low_res = self.transform_low(high_res)
        high_res = self.normalize(high_res)

        return low_res, high_res

    def __len__(self):
        """
        returns the length files for this dataset class
        :return:
        """
        return len(self.files)
