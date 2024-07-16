import os
from PIL import Image
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, lidar_paths = self.get_paths(opt)

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.lidar_paths = lidar_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_dir = opt.label_dir
        image_dir = opt.image_dir
        lidar_dir = opt.lidar_dir

        label_paths = sorted(make_dataset(label_dir, opt.max_dataset_size))
        image_paths = sorted(make_dataset(image_dir, opt.max_dataset_size))
        lidar_paths = sorted(make_dataset(lidar_dir, opt.max_dataset_size))

        return label_paths, image_paths, lidar_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image (Depth)
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')  # Load as grayscale
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # Input Image (RGB)
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # Lidar Scan (Grayscale)
        lidar_path = self.lidar_paths[index]
        lidar = Image.open(lidar_path).convert('L')  # Load as grayscale
        transform_lidar = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        lidar_tensor = transform_lidar(lidar) * 255.0

        # Instance Map (if needed)
        if self.opt.no_instance:
            instance_tensor = None  # or any placeholder you choose
        else:
            instance_path = self.instance_paths[index]  # Adjust as per your dataset structure
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        # Combine into input dictionary
        input_dict = {
            'label': label_tensor,
            'image': image_tensor,
            'lidar': lidar_tensor,
            'instance': instance_tensor,  # Include instance map if needed
            'label_path': label_path,
            'image_path': image_path,
            'lidar_path': lidar_path
        }

        # Postprocess if needed (subclasses can override)
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        # Example of postprocessing if needed
        pass

    def __len__(self):
        return self.dataset_size
