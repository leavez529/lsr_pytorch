import numpy as np
import torch
import logging

def load_nlcd_stats(
    stats_mu="data/nlcd_mu.txt",
    stats_sigma="data/nlcd_sigma.txt",
    class_weights="data/nlcd_class_weights.txt",
    lr_classes=22,
    hr_classes=5,
    tensor=False
):
    stats_mu = np.loadtxt(stats_mu)
    assert lr_classes == stats_mu.shape[0]
    assert hr_classes == (stats_mu.shape[1] + 1)
    nlcd_means = np.concatenate([np.zeros((lr_classes, 1)), stats_mu], axis=1)
    nlcd_means[nlcd_means == 0] = 0.000001
    nlcd_means[:, 0] = 0
    nlcd_means = do_nlcd_means_tuning(nlcd_means)

    stats_sigma = np.loadtxt(stats_sigma)
    assert lr_classes == stats_sigma.shape[0]
    assert hr_classes == (stats_sigma.shape[1] + 1)
    nlcd_vars = np.concatenate([np.zeros((lr_classes, 1)), stats_sigma], axis=1)
    nlcd_vars[nlcd_vars < 0.0001] = 0.0001

    if not class_weights:
        nlcd_class_weights = np.ones((lr_classes,))
    else:
        nlcd_class_weights = np.loadtxt(class_weights)
        assert lr_classes == nlcd_class_weights.shape[0]

    if tensor:
        nlcd_class_weights = torch.from_numpy(nlcd_class_weights)
        nlcd_means = torch.from_numpy(nlcd_means)
        nlcd_vars = torch.from_numpy(nlcd_vars)

    return nlcd_class_weights, nlcd_means, nlcd_vars

def do_nlcd_means_tuning(nlcd_means):
    nlcd_means[2:, 1] -= 0
    nlcd_means[3:7, 4] += 0.25
    nlcd_means = nlcd_means / np.maximum(0, nlcd_means).sum(axis=1, keepdims=True)
    nlcd_means[0, :] = 0
    nlcd_means[-1, :] = 0
    return nlcd_means

def to_float(arr, data_type="int8"):
    if data_type == "int8":
        res = np.clip(arr / 255.0, 0.0, 1.0)
    elif data_type == "int16":
        res = np.clip(arr / 4096.0, 0.0, 1.0)
    else:
        raise ValueError("Select an appropriate data type.")
    return res


def classes_in_key(key_txt):
    key_array = np.loadtxt(key_txt)
    return len(np.unique(key_array[:, 1]))


def handle_labels(arr, key_txt):
    key_array = np.loadtxt(key_txt)
    trans_arr = arr

    for translation in key_array:
        # translation is (src label, dst label)
        scr_l, dst_l = translation
        if scr_l != dst_l:
            trans_arr[trans_arr == scr_l] = dst_l

    # translated array
    return trans_arr

def get_logger(logger_name, log_file, level=logging.INFO):
        logger = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)

def color_aug(colors):
    n_ch = colors.shape[-1]
    contra_adj = 0.50
    bright_adj = 0.50

    ch_mean = np.mean(colors, axis=(0, 1), keepdims=True).astype(np.float32)

    contra_mul = np.random.uniform(1 - contra_adj, 1 + contra_adj, (1, 1, n_ch)).astype(
        np.float32
    )
    bright_mul = np.random.uniform(1 - bright_adj, 1 + bright_adj, (1, 1, n_ch)).astype(
        np.float32
    )

    colors = (colors - ch_mean) * contra_mul + ch_mean * bright_mul
    return colors