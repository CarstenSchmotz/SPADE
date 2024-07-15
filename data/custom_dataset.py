import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from data.base_dataset import BaseDataset
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

        # Read RGB image and depth (label) as grayscale
        image = Image.open(image_path).convert('RGB')  # RGB image
        label = Image.open(label_path).convert('L')    # Depth image (grayscale)
        lidar = Image.open(lidar_path).convert('L')    # Lidar scan (grayscale)

        # Combine RGB and depth into RGBD (4-channel)
        rgb_np = np.array(image)
        depth_np = np.array(label)
        rgbd_image = np.concatenate((rgb_np, depth_np[:, :, np.newaxis]), axis=2)

        # Apply transformations
        params = get_params(self.opt, label.size)
        transform_rgbd = get_transform(self.opt, params, grayscale=False)  # Assuming get_transform handles 4-channel input
        transform_lidar = get_transform(self.opt, params, grayscale=True)  # Adjust for lidar input

        rgbd_image = transform_rgbd(rgbd_image)
        lidar = transform_lidar(lidar)

        return {'rgbd': rgbd_image, 'lidar': lidar, 'label_path': label_path, 'image_path': image_path, 'lidar_path': lidar_path}

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
