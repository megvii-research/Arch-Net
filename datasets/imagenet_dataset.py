#!/usr/bin/env python
# coding=utf-8
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class CustomImagenetDataset(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = datasets.folder.default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            num_images: int = 0,
    ):
        super(CustomImagenetDataset, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        if num_images > 0:
            random.shuffle(self.samples)
            self.samples = self.samples[:num_images]

