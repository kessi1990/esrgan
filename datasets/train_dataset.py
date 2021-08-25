from torch.utils.data.dataset import Dataset
from PIL import Image

from utils import utils_img


class TrainDataset(Dataset):
    """

    """
    def __init__(self, root, image_size=128, upscale_factor=4):
        """

        :param root:
        :param image_size:
        :param upscale_factor:
        """
        super(TrainDataset, self).__init__()

        low_res_size = image_size // upscale_factor

        self.files = utils_img.load_images(root)

        self.transform_low = utils_img.transform_low(low_res_size)

        self.transform_high = utils_img.transform_high(image_size)

        self.normalize = utils_img.normalize()

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        file = Image.open(self.files[index]).convert('RGB')

        high_res = self.transform_high(file)

        low_res = self.transform_low(high_res)

        high_res = self.normalize(high_res)

        return low_res, high_res

    def __len__(self):
        """

        :return:
        """
        return len(self.files)
