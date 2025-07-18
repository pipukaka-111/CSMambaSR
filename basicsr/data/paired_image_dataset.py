from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.
    读取 LQ（低质量，例如低分辨率、模糊、噪声等）和 GT 图像对。

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.
    有三种模式：
    1、'lmdb': 使用 lmdb 文件。
        如果 opt['io_backend'] == 'lmdb'。
    2、'meta_info_file': 使用元信息文件生成路径。
        如果 opt['io_backend'] != 'lmdb' 且 opt['meta_info_file'] 不为 None。
    3、'folder': 扫描文件夹生成路径。
        其余情况。

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
            
        opt (dict): 训练数据集的配置，包含以下键：
            dataroot_gt (str): GT 数据的根路径。
            dataroot_lq (str): LQ 数据的根路径。
            meta_info_file (str): 元信息文件的路径。
            io_backend (dict): IO 后端类型及其他关键字参数。
            filename_tmpl (str): 每个文件名的模板。注意模板不包括文件扩展名。
                默认值：'{}'。
            gt_size (int): GT 补丁的裁剪大小。
            use_hflip (bool): 是否使用水平翻转。
            use_rot (bool): 是否使用旋转（使用垂直翻转并转置高度和宽度进行实现）。
            scale (bool): 缩放，将自动添加。
            phase (str): 'train' 或 'val'。
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)

        # file client (io backend) 这是初始化 file client 部分
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # 赋值常用的配置参数
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0
        
        # 这块内容是根据 GT 和 LQ 的图像目录读取出相应的文件列表
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        # 如果输入是lmdb，则使用paired_paths_from_lmdb函数
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # 如果输入是meta_info_file方式，则使用paired_paths_from_meta_info_file函数
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        # 如果是一般文件目录方式，则使用paired_paths_from_folder函数
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl, self.task)
        # 以上这些函数都已经实现在basicsr/data/data_util.py文件中

    def __getitem__(self, index):
        '''定义了从输入图像，经过变换、数据增强等变为PyTorchTensor的过程。

        Args:
            index (_type_): _description_

        Returns:
            _type_: _description_
        '''
        # 初始化file client
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        
        # 下面这个代码块是从存储介质中读取相应的数据到内存的过程
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1

            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
    
        elif self.task == 'denoising_gray': # Matlab + OpenCV version
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            # OpenCV version, following "Deep Convolutional Dictionary Learning for Image Denoising"
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            # # Matlab version (using this version may have 0.6dB improvement, which is unfair for comparison)
            # img_gt = imfrombytes(img_bytes, flag='unchanged', float32=True)
            # if img_gt.ndim != 2:
            #     img_gt = rgb2ycbcr(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), y_only=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise/255., img_gt.shape)

        else:
            
            # image range: [0, 1], float32., H W 3
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training 训练阶段的数据增强（旋转 裁切等）
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform 可进行色彩空间转换
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # 以下代码块将 numpy 数据格式转换成 PyTorch 所需的 Tensor 格式，并根据需要作归一化
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # 最后，我们返回一个字典，包括输入的 LQ 图像，作为标签的 GT 图像，以及他们的路径
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
