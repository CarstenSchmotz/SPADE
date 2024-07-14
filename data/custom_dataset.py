"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """
    # In data/custom_dataset.py

import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
# custom_dataset.py

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomDataset(BaseDataset):
    def __init__(self, opt):
        """Initialize the custom dataset."""
        BaseDataset.__init__(self)  # Initialize base class
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
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        label_path = self.label_paths[index]
        image_path = self.image_paths[index]
        lidar_path = self.lidar_paths[index]
        label = Image.open(label_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        lidar = Image.open(lidar_path).convert('RGB')
        label = self.transform(label)
        image = self.transform(image)
        lidar = self.transform(lidar)
        return {'label': label, 'image': image, 'lidar': lidar,
                'label_path': label_path, 'image_path': image_path, 'lidar_path': lidar_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.label_paths)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add specific options for this dataset."""
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--lidar_dir', type=str, required=True,
                            help='path to the directory that contains lidar images')
        parser.add_argument('--val_label_dir', type=str, required=True,
                            help='path to the directory that contains validation label images')
        parser.add_argument('--val_image_dir', type=str, required=True,
                            help='path to the directory that contains validation photo images')
        parser.add_argument('--val_lidar_dir', type=str, required=True,
                            help='path to the directory that contains validation lidar images')
        return parser


    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths
