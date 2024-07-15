import random
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset

class CustomDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.label_dir = opt.label_dir
        self.image_dir = opt.image_dir
        self.lidar_dir = opt.lidar_dir
        self.val_label_dir = opt.val_label_dir
        self.val_image_dir = opt.val_image_dir
        self.val_lidar_dir = opt.val_lidar_dir

        self.label_paths = sorted(make_dataset(self.label_dir, opt.max_dataset_size))
        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))
        self.lidar_paths = sorted(make_dataset(self.lidar_dir, opt.max_dataset_size))
        self.val_label_paths = sorted(make_dataset(self.val_label_dir, opt.max_dataset_size))
        self.val_image_paths = sorted(make_dataset(self.val_image_dir, opt.max_dataset_size))
        self.val_lidar_paths = sorted(make_dataset(self.val_lidar_dir, opt.max_dataset_size))

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        image_path = self.image_paths[index]
        lidar_path = self.lidar_paths[index]

        # Read images
        label = Image.open(label_path).convert('L')  # Convert to grayscale
        image = Image.open(image_path).convert('RGB')  # RGB
        lidar = Image.open(lidar_path).convert('L')  # Convert to grayscale

        # Get transformation parameters
        params = get_params(self.opt, label.size)  # Assuming label.size as the size of the image

        # Apply transformations
        transform_label = get_transform(self.opt, params, grayscale=True)
        transform_image = get_transform(self.opt, params)
        transform_lidar = get_transform(self.opt, params, grayscale=True)

        label = transform_label(label)
        image = transform_image(image)
        lidar = transform_lidar(lidar)

        return {'label': label, 'image': image, 'lidar': lidar, 'label_path': label_path, 'image_path': image_path, 'lidar_path': lidar_path}

    def __len__(self):
        return len(self.label_paths)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--label_dir', type=str, required=True, help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True, help='path to the directory that contains photo images')
        parser.add_argument('--lidar_dir', type=str, required=True, help='path to the directory that contains lidar images')
        parser.add_argument('--val_label_dir', type=str, required=True, help='path to the directory that contains validation label images')
        parser.add_argument('--val_image_dir', type=str, required=True, help='path to the directory that contains validation photo images')
        parser.add_argument('--val_lidar_dir', type=str, required=True, help='path to the directory that contains validation lidar images')
        return parser
