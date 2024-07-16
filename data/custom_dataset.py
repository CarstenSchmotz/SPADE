import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.base_dataset import get_params, get_transform 
from torchvision import transforms

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

        try:
            # Read RGB image and depth (label) as grayscale
            image = Image.open(image_path).convert('RGB')  # RGB image
            label = Image.open(label_path).convert('L')    # Depth image (grayscale)
            lidar = Image.open(lidar_path).convert('L')    # Lidar scan (grayscale)

            # Convert images to numpy arrays
            rgb_np = np.array(image)
            depth_np = np.array(label)
            lidar_np = np.array(lidar)

            # Combine RGB and depth into RGBD (4-channel)
            rgbd_image = np.concatenate((rgb_np, depth_np[:, :, np.newaxis]), axis=2)

            # Print shapes for debugging
            print(f"RGB shape: {rgb_np.shape}, Depth shape: {depth_np.shape}, Lidar shape: {lidar_np.shape}")
            print(f"RGBD image shape: {rgbd_image.shape}")

            # Apply transformations
            #print(label, "label",label.size, "label.size")
            params = get_params(self.opt, label.size)
            transform_rgbd = self.get_transform_rgbd(params)
            transform_lidar = self.get_transform_lidar(params)

            rgbd_image = transform_rgbd(Image.fromarray(rgbd_image))
            lidar = transform_lidar(Image.fromarray(lidar_np))
            print(f"Transformed RGBD shape: {rgbd_image.shape}, Transformed Lidar shape: {lidar.shape}")

            return {'rgbd': rgbd_image, 'lidar': lidar, 'label_path': label_path, 'image_path': image_path, 'lidar_path': lidar_path}
        
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            raise

    

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
