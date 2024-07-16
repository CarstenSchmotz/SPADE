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
            params = get_params(self.opt, label.size)
            transform_rgbd = self.get_transform_rgbd(params, input_nc=4)
            transform_lidar = self.get_transform_lidar(params, input_nc=1)

            rgbd_image = transform_rgbd(Image.fromarray(rgbd_image))
            lidar = transform_lidar(Image.fromarray(lidar_np))
            print(f"Transformed RGBD shape: {rgbd_image.shape}, Transformed Lidar shape: {lidar.shape}")

            return {'rgbd': rgbd_image, 'lidar': lidar, 'label_path': label_path, 'image_path': image_path, 'lidar_path': lidar_path}
        
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            raise

    def get_transform_rgbd(self, params, method=Image.BICUBIC, normalize=True, toTensor=True):
        transform_list = []

        if 'resize' in self.opt.preprocess_mode:
            osize = [self.opt.load_size, self.opt.load_size]
            transform_list.append(transforms.Resize(osize, interpolation=method))
        elif 'scale_width' in self.opt.preprocess_mode:
            transform_list.append(transforms.Lambda(lambda img: self.__scale_width(img, self.opt.load_size, method)))
        elif 'scale_shortside' in self.opt.preprocess_mode:
            transform_list.append(transforms.Lambda(lambda img: self.__scale_shortside(img, self.opt.load_size, method)))

        if 'crop' in self.opt.preprocess_mode:
            transform_list.append(transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], self.opt.crop_size)))

        if self.opt.preprocess_mode == 'none':
            base = 32
            transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base, method)))

        if self.opt.preprocess_mode == 'fixed':
            w = self.opt.crop_size
            h = round(self.opt.crop_size / self.opt.aspect_ratio)
            transform_list.append(transforms.Lambda(lambda img: self.__resize(img, w, h, method)))

        if self.opt.isTrain and not self.opt.no_flip:
            transform_list.append(transforms.Lambda(lambda img: self.__flip(img, params['flip'])))

        if toTensor:
            transform_list.append(transforms.ToTensor())

        if normalize:
            if self.opt.input_nc == 4:  # RGBD input
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)

    def get_transform_lidar(self, params, method=Image.BICUBIC, normalize=True, toTensor=True):
        transform_list = []

        if 'resize' in self.opt.preprocess_mode:
            osize = [self.opt.load_size, self.opt.load_size]
            transform_list.append(transforms.Resize(osize, interpolation=method))
        elif 'scale_width' in self.opt.preprocess_mode:
            transform_list.append(transforms.Lambda(lambda img: self.__scale_width(img, self.opt.load_size, method)))
        elif 'scale_shortside' in self.opt.preprocess_mode:
            transform_list.append(transforms.Lambda(lambda img: self.__scale_shortside(img, self.opt.load_size, method)))

        if 'crop' in self.opt.preprocess_mode:
            transform_list.append(transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], self.opt.crop_size)))

        if self.opt.preprocess_mode == 'none':
            base = 32
            transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base, method)))

        if self.opt.preprocess_mode == 'fixed':
            w = self.opt.crop_size
            h = round(self.opt.crop_size / self.opt.aspect_ratio)
            transform_list.append(transforms.Lambda(lambda img: self.__resize(img, w, h, method)))

        if self.opt.isTrain and not self.opt.no_flip:
            transform_list.append(transforms.Lambda(lambda img: self.__flip(img, params['flip'])))

        if toTensor:
            transform_list.append(transforms.ToTensor())

        if normalize:
            if self.opt.input_nc == 1:  # Lidar input (assuming grayscale)
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))

        return transforms.Compose(transform_list)

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
