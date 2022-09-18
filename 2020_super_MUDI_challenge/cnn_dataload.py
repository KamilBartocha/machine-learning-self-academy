import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os


class DiffusionDataset(Dataset):

    def __init__(self, data_path_in, data_path_out):
        super(DiffusionDataset, self).__init__()

        # get filenames from a directory
        self.filenames = self.__get_file_list(data_path_in)
        self.filenames_out = self.__get_file_list(data_path_out)

        # input data
        self.input_transform = transforms.Compose([transforms.ToTensor()])

        # output data
        self.output_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        # load files (input, target)
        input = self.__load_numpy_image(self.filenames[index])
        target = self.__load_numpy_image(self.filenames_out[index])

        input = self.input_transform(input)
        target = self.output_transform(target)

        return input, target

    def __len__(self):
        return len(self.filenames)

    def __get_file_list(self, data_path):
        filenames_input = os.listdir(data_path)
        filenames = []

        for file in filenames_input:
            if self.__is_numpy_file(data_path + file):
                filenames.append(data_path + file)

        return filenames

    def __load_numpy_image(self, filename):
        image = np.load(filename)
        return image

    def __is_numpy_file(self, filename):
        return filename.endswith('npy')

