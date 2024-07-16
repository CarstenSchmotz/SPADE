import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

class CustomRGBDLidarDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_rgbd = os.path.join(opt.dataroot, 'rgbd')
        self.dir_lidar = os.path.join(opt.dataroot, 'lidar')
        self.rgbd_paths = sorted(make_dataset(self.dir_rgbd))
        self.lidar_paths = sorted(make_dataset(self.dir_lidar))

    def __getitem__(self, index):
        rgbd_path = self.rgbd_paths[index]
        lidar_path = self.lidar_paths[index]

        rgbd = Image.open(rgbd_path).convert('RGB')
        depth = Image.open(rgbd_path.replace('.jpg', '_depth.png')).convert('L') # Assuming depth images are saved with `_depth` suffix
        rgbd = np.concatenate((np.array(rgbd), np.array(depth)[:, :, np.newaxis]), axis=2) # Combine RGB and depth

        lidar = Image.open(lidar_path).convert('L')

        transform_params = get_params(self.opt, rgbd.shape)
        transform_rgbd = get_transform(self.opt, transform_params, normalize=False)
        transform_lidar = get_transform(self.opt, transform_params, normalize=False)

        rgbd = transform_rgbd(rgbd)
        lidar = transform_lidar(lidar)

        return {'A': rgbd, 'B': lidar, 'A_paths': rgbd_path, 'B_paths': lidar_path}

    def __len__(self):
        return len(self.rgbd_paths)

    def name(self):
        return 'CustomRGBDLidarDataset'
