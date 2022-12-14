import torch
import numpy as np
import logging
import os
import pandas as pd
import torch.nn.functional as F
from util import handle_labels, classes_in_key, do_nlcd_means_tuning, to_float, load_nlcd_stats, color_aug
import albumentations as albu

def get_weak_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
    ]
    return albu.Compose(train_transform)

class LandCoverDataset(Dataset):
    def __init__(
        self,
        data_dir,
        img_shape,
        states,
        split="train",
        num_classes=5,
        lr_num_classes=22,
        hr_labels_index=8,
        lr_labels_index=9,
        hr_label_key="data/cheaseapeake_to_hr_labels.txt",
        lr_label_key="data/nlcd_to_lr_labels.txt",
        do_color_aug=False,
        do_superres=False,
        data_type="int8",
        aug=None,
    ):
        self.data_dir = data_dir
        self.img_shape = img_shape

        self.num_classes = num_classes
        self.lr_num_classes = lr_num_classes

        self.do_color_aug = do_color_aug
        self.do_superres = do_superres
        self.data_type = data_type

        self.hr_labels_index = hr_labels_index
        self.lr_labels_index = lr_labels_index
        self.hr_label_key = hr_label_key
        self.lr_label_key = lr_label_key

        self.aug = aug
        if self.hr_label_key:
            assert self.num_classes == classes_in_key(self.hr_label_key)
        if self.lr_label_key:
            assert self.lr_num_classes == classes_in_key(self.lr_label_key)

        self.patches = []

        for state in states:
            logging.info("Adding %s patches from %s" % (split, state))
            fn = os.path.join(self.data_dir, "%s_extended-%s_patches.csv" % (state, split))
            if not os.path.isfile(fn):
                fn = os.path.join(self.data_dir, "%s-%s_patches.csv" % (state, split))
            df = pd.read_csv(fn)
            for row in df.itertuples():
                fn = getattr(row, "patch_fn")
                index = getattr(row, "patch_id")
                self.patches.append((os.path.join(self.data_dir, fn), state, index))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        fn, state, idx = self.patches[index]
        if fn.endswith(".npz"):
            dl = np.load(fn)
            data = dl["arr_0"].squeeze()
            dl.close()
        elif fn.endswith(".npy"):
            data = np.load(fn).squeeze()
        data = np.rollaxis(data, 0, 3)

        # the patch should be a square
        assert data.shape[0] == data.shape[1]
        data_size = data.shape[0]
        input_size = self.img_shape[0]

        # do a random crop if input_size is less than the prescribed size
        if input_size < data_size:
            x_idx = np.random.randint(0, data_size - input_size)
            y_idx = np.random.randint(0, data_size - input_size)
            data = data[
                y_idx : y_idx + input_size, x_idx : x_idx + input_size, :
            ]

        # get the image
        img = to_float(data[:, :, : self.img_shape[2]], self.data_type)

        # get y_highres
        if self.hr_label_key:
            y_hr = handle_labels(
                data[:, :, self.hr_labels_index], self.hr_label_key
            )
        else:
            y_hr = data[:, :, self.hr_labels_index]

        # do augmentation
        if self.aug:
            sample_weak = self.aug[0](image=img, masks=y_hr)
            img, y_hr = sample_weak["image"], sample_weak["masks"]

        # to torch
        img = torch.from_numpy(img).type(torch.FloatTensor)
        y_hr = torch.from_numpy(y_hr).type(torch.LongTensor)

        # HWC to CHW
        img = img.permute((2, 0, 1))

        # if do super resolution, get y_nlcd
        if self.do_superres:
            if self.lr_label_key:
                y_nlcd = handle_labels(
                    data[:, :, self.lr_labels_index], self.lr_label_key
                )
            else:
                y_nlcd = data[:, :, self.lr_labels_index]
            y_nlcd = torch.from_numpy(y_nlcd).type(torch.LongTensor)

        if self.do_superres:
            return {
                "image": img,
                "label_hr": y_hr,
                "label_nlcd": y_nlcd,
            }
        else:
            return {
                "image": img,
                "label_hr": y_hr,
            }