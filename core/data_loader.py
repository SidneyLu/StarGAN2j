from pathlib import Path
from itertools import chain
import os
import random

import jittor.distributions
from munch import Munch
from PIL import Image
import numpy as np

import jittor
from jittor import transform
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
from jittor.dataset import Sampler

"""This is the module for data loading"""
"""数据加载模块"""


#Filtering images
#筛选出图像文件
def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

#默认数据集
class DefaultDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.samples = []
        for f in os.listdir(root):
            self.samples.append(os.path.join(root, f))

        self.samples.sort()
        self.targets = None
        self.set_attrs(
            batch_size=1,
            total_len=len(self.samples),
            shuffle=False
        )

        print(f"数据集初始化完成: {root}, 样本数: {len(self.samples)}")

    def __getitem__(self, index):
        img_path = self.samples[index]

        try:
            img = Image.open(img_path).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
                img = jittor.array(np.array(img))

            # 确保张量是3D [C, H, W]
            assert img.ndim == 3, f"样本维度错误: 期望3D，实际{img.ndim}D"

            return img

        except Exception as e:
            print(f"错误: 无法加载图像 {img_path}: {str(e)}")
            # 返回占位张量
            return jittor.zeros((3, 256, 256))

#参考数据集
class ReferenceDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    #将图像连同标签整合成网络的输入
    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)

#Customized ImageFolder, as the substitute of torchvision.utils.ImageFolder
#自定义的ImageFolder，为了对齐TorchVision中同名模块
class ImageFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.classes = self._find_classes(root)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset(root, self.class_to_idx)
        self.targets = [s[1] for s in self.samples]

        self.total_len = len(self.samples)

    def _find_classes(self, root):
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.sort()  # 确保类别顺序固定
        if len(classes) == 0:
            raise FileNotFoundError(f"根目录 {root} 下未找到子文件夹（类别）")
        return classes

    def _make_dataset(self, root, class_to_idx):
        samples = []
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue  # 跳过非文件夹
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    samples.append((img_path, cls_idx))
        if len(samples) == 0:
            raise FileNotFoundError(f"未在 {root} 下找到图像文件")
        return samples

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = Image.open(img_path).convert('RGB')  # 统一转换为 RGB 格式
        img = jittor.array(np.array(img))  # 转换为 Jittor 张量
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return self.total_len

#Customized Sampler, as the substitute of torch.dataset.WeightedRandomSampler
#自定义的采样器，为了对齐torch.dataset中同名模块
class WeightedRandomSampler(Sampler):
    def __init__(self, dataset, weights, num_samples, replacement=False, generator=None,):
        super().__init__(dataset)
        self.dataset = dataset
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rand_tensor = jittor.multinomial(self.weights, self.num_samples, self.replacement)
        yield from iter(rand_tensor.numpy().tolist())

    def __len__(self):
        return self.num_samples

#Calculate the weight of each class for random sampling
#根据各类样本数计算相应的权重，进行随机取样
def _make_balanced_sampler(dataset, labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(dataset, weights, len(weights))

#Crop source images randomly, use the probability as threshold
#按一定概率阈值裁切图像以进行数据增强
def random_crop_transform(x, crop_fn, probability):
    if random.random() < probability:
        return crop_fn(x)
    else:
        return x

#Implementing the same function as the RandomCropTransform from official PyTorch Implementation
#对齐原仓库中的同名模块
class RandomCropTransform:
    def __init__(self, crop_fn, probability):
        self.crop_fn = crop_fn
        self.probability = probability

    def __call__(self, x):
        return random_crop_transform(x, self.crop_fn, self.probability)

#Load data for training
#加载训练数据
def get_train_loader(root, which='source', img_size=256,
                     batch_size=4, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    crop = transform.RandomResizedCrop(img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = RandomCropTransform(crop,prob)

    transforms = transform.Compose([
        rand_crop,
        transform.Resize([img_size, img_size]),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if which == 'source':
        dataset = ImageFolder(root, transforms)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transforms)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset, dataset.targets)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      num_workers=num_workers,
                      drop_last=True)

#Load data for evaluation
#加载评估数据
def get_eval_loader(root, img_size=256, batch_size=4,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print(f'准备评估数据加载器... 根目录: {root}')

    # 确定标准化参数
    # 适用于InceptionV3
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transforms = transform.Compose([
        transform.Resize([height, width]),
        transform.ToTensor(),
        transform.ImageNormalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transforms)
    print(f"数据集大小: {len(dataset)}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    try:
        batch = next(iter(loader))
        print(f"测试批次: 形状={batch.shape}, 维度={batch.ndim}")
        assert batch.ndim == 4, f"批次维度错误: 期望4D，实际{batch.ndim}D"
    except Exception as e:
        print(f"测试批次加载失败: {e}")

    return loader

#Load data for testing
#加载测试数据
def get_test_loader(root, img_size=256, batch_size=4,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transforms = transform.Compose([
        transform.Resize([img_size, img_size]),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transforms)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      )
#For input iteration
#输入读取器
class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.mode = mode
        self.iter = iter(self.loader)

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = jittor.randn(x.shape[0], self.latent_dim)
            z_trg2 = jittor.randn(x.shape[0], self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v
                      for k, v in inputs.items()})


