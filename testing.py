import pandas as pd
import logging
import datetime
import time
import os
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from util import to_float, handle_labels, load_nlcd_stats
from evaluate import compute_accuracy, compute_jaccard, developed_accuracy, developed_jaccard, multiclass_jaccard
from pathlib import Path
import segmentation_models_pytorch as smp


class Test:
    def __init__(
        self,
        data_dir,
        input_fn: str,
        output_dir,
        model_file,
        device,
        input_size=240,
        n_channels=4,
        n_classes=5,
        batch_size=16,
        superres=False,
        data_type="int8",
    ):
        self.data_dir = data_dir
        self.input_fn = input_fn
        self.output_dir = output_dir
        self.model_file = model_file
        self.input_size = input_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.superres = superres
        self.data_type = data_type
        self.device = device
        self.batch_size = batch_size

    def load_model(self):
        net = smp.Unet(
            encoder_name="se_resnext50_32x4d",
            encoder_weights=None,
            decoder_use_batchnorm=True,
            decoder_attention_type=None,
            in_channels=self.n_channels,
            classes=self.n_classes,
        )
        net.load_state_dict(torch.load(self.model_file, map_location=self.device))
        net.to(device=self.device)
        return net

    def load_tiles(self):
        try:
            df = pd.read_csv(self.input_fn)
            fns = df[["naip-new_fn", "lc_fn", "nlcd_fn"]].values
            return fns
        except Exception as e:
            logging.error("Could not load the input file")
            logging.error(e)
            return

    @staticmethod
    def run_model_on_tile(
        model,
        naip_tile: np.array,
        inpt_size: int,
        output_size: int,
        batch_size: int,
        device,
    ):
        down_weight_padding = 40
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        stride_x = inpt_size - down_weight_padding * 2
        stride_y = inpt_size - down_weight_padding * 2
        output = torch.zeros(height, width, output_size).type(torch.FloatTensor).to(device=device)
        counts = torch.zeros(height, width).type(torch.FloatTensor).to(device=device) + 0.000000001
        kernel = torch.ones(inpt_size, inpt_size).type(torch.FloatTensor).to(device=device) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[
            down_weight_padding : down_weight_padding + stride_y,
            down_weight_padding : down_weight_padding + stride_x,
        ] = 5

        patches = []
        y_indices = []
        x_indices = []

        for y_index in list(range(0, height - inpt_size, stride_y)) + [
            height - inpt_size,
        ]:
            for x_index in list(range(0, width - inpt_size, stride_x)) + [
                width - inpt_size,
            ]:
                naip_im = naip_tile[
                    y_index : y_index + inpt_size, x_index : x_index + inpt_size, :
                ]

                patches.append(naip_im)
                y_indices.append(y_index)
                x_indices.append(x_index)

        data = TestDataset(patches, y_indices, x_indices)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

        for batch in loader:
            mini_patches = batch["patch"]
            mini_y_indices = batch["y_index"]
            mini_x_indices = batch["x_index"]
            mini_patches = torch.FloatTensor(mini_patches).permute((0, 3, 1, 2)).to(device=device)
            with torch.no_grad():
                mask_pred = model(mini_patches)
            model_output = F.softmax(mask_pred, dim=1).permute((0, 2, 3, 1))
            for i in range(0, len(mini_patches)):
                y = mini_y_indices[i]
                x = mini_x_indices[i]
                output[y : y + inpt_size, x : x + inpt_size] += (
                    model_output[i] * kernel[..., np.newaxis]
                )
                counts[y : y + inpt_size, x : x + inpt_size] += kernel

        return output / counts[..., np.newaxis]


    def run_on_tiles(self, get_color = True):
        logging.info(
            "Starting %s at %s"
            % ("Model inference script", str(datetime.datetime.now()))
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.start_time = float(time.time())

        fns = self.load_tiles()
        model = self.load_model()
        model.eval()
        accu = 0
        accu_class = 0
        accu_d = 0
        iou = 0
        iou_class = 0
        iou_d = 0
        for i in range(len(fns)):
            tic = float(time.time())
            naip_fn = os.path.join(self.data_dir, fns[i][0])
            lc_fn = os.path.join(self.data_dir, fns[i][1])
            nlcd_fn = os.path.join(self.data_dir, fns[i][2])

            logging.info("Running model on %s\t%d/%d" % (naip_fn, i + 1, len(fns)))

            naip_fid = rasterio.open(naip_fn, "r")
            naip_profile = naip_fid.meta.copy()
            naip_tile = to_float(naip_fid.read().astype(np.float32), self.data_type)
            naip_tile = np.rollaxis(naip_tile, 0, 3)
            naip_fid.close()

            output = self.run_model_on_tile(
                model,
                naip_tile,
                self.input_size,
                self.n_classes,
                self.batch_size,
                self.device,
            )
            output = output[:, :, : self.n_classes]

            lc_fid = rasterio.open(lc_fn, "r")
            lc_tile = lc_fid.read().astype(np.int64)
            lc_fid.close()
            lc_tile = handle_labels(lc_tile, "./data/cheaseapeake_to_hr_labels.txt")
            if self.n_classes == 4:
                lc_tile -= 1

            nlcd_fid = rasterio.open(nlcd_fn, "r")
            nlcd_tile = nlcd_fid.read().astype(np.int32)
            nlcd_tile = np.squeeze(nlcd_tile, axis=0)
            nlcd_fid.close()
            nlcd_tile = handle_labels(nlcd_tile, "./data/nlcd_to_lr_labels.txt")

            output_classes = output.argmax(dim=2)
            output_class_fn = os.path.basename(naip_fn)[:-4] + "_class.tif"

            output_hot = F.one_hot(output_classes, self.n_classes)
            output_hot = torch.unsqueeze(output_hot, 0)
            output_hot = output_hot.permute((0, 3, 1, 2))
            lc_hot = F.one_hot(torch.from_numpy(lc_tile), self.n_classes)
            lc_hot = lc_hot.permute((0, 3, 1, 2)).to(device=self.device)

            i, i_class = multiclass_jaccard(output_hot.float(), lc_hot.float(), class_score=True)
            a, a_class = compute_accuracy(output_classes, torch.from_numpy(lc_tile).long(), True)
            i, i_class = compute_jaccard(output_classes, torch.from_numpy(lc_tile).long(), True)
            a_d = developed_accuracy(output_classes, torch.from_numpy(lc_tile).long(), torch.from_numpy(nlcd_tile).long())
            i_d = developed_jaccard(output_classes, torch.from_numpy(lc_tile).long(), torch.from_numpy(nlcd_tile).long())

            accu += a
            accu_class += a_class
            iou += i
            iou_class += i_class
            accu_d += a_d
            iou_d += i_d

            current_profile = naip_profile.copy()
            current_profile["driver"] = "GTiff"
            current_profile["dtype"] = "uint8"
            current_profile["count"] = 1
            current_profile["compress"] = "lzw"

            logging.info("Finished iteration in %0.4f seconds" % (time.time() - tic))

            # generate prediction color maps
            if get_color:
                output_color_fn = os.path.basename(naip_fn)[:-4] + "_class_color.tif"
                self.classes2color(output_classes, self.n_classes, os.path.join(self.output_dir, output_color_fn), naip_profile, "data/hr_color.txt")
        self.end_time = float(time.time())
        accu = accu / len(fns)
        accu_class = accu_class / len(fns)
        iou = iou / len(fns)
        iou_class = iou_class / len(fns)
        accu_d = accu_d / len(fns)
        iou_d = iou_d / len(fns)
        print("accuracy: ", accu, accu_class)
        print("IoU: ", iou, iou_class)
        print("developed accuracy: ", accu_d)
        print("developed IoU: ", iou_d)
        logging.info(
            "Finished %s in %0.4f seconds"
            % ("Model inference script", self.end_time - self.start_time)
        )

    @staticmethod
    def classes2color(output_classes, n_classes, output_fn, naip_profile, color_fn):
        classes = F.one_hot(output_classes, n_classes).cpu().numpy()
        color_file = pd.read_csv(color_fn, sep=', ')
        color_matrix = color_file.loc[:,["red","green","blue"]].values
        color = np.matmul(classes, color_matrix)
        color = np.rollaxis(color, axis=2)

        current_profile = naip_profile.copy()
        current_profile["driver"] = "GTiff"
        current_profile["dtype"] = "uint8"
        current_profile["count"] = 3
        current_profile["compress"] = "lzw"

        f = rasterio.open(
            output_fn, "w", **current_profile
        )
        f.write(color)
        f.close()

    @staticmethod
    def compute_accuracy(pred, mask_true):
        equal_map = np.equal(pred, mask_true)
        num = np.sum(equal_map)
        denom = mask_true.size
        return num / denom

    def gen_true_color(self):
        fns = self.load_tiles()
        for i in range(len(fns)):
            lc_fn = os.path.join(self.data_dir, fns[i][1])
            lc_fid = rasterio.open(lc_fn, "r")
            lc_profile = lc_fid.meta.copy()
            lc_tile = lc_fid.read().astype(np.int64)
            lc_tile = np.squeeze(lc_tile, axis=0)
            lc_fid.close()

            lc_tile = handle_labels(lc_tile, "./data/cheaseapeake_to_hr_labels.txt")
            output = F.one_hot(torch.from_numpy(lc_tile), self.n_classes).numpy()
            output_color_fn = os.path.basename(lc_fn)[:-4] + "_class_color.tif"
            self.classes2color(lc_tile, os.path.join(self.output_dir, output_color_fn), lc_profile, "data/hr_color.txt")

class TestDataset(Dataset):
    def __init__(
        self,
        patches,
        y_indices,
        x_indices,
    ):
        self.patches = patches
        self.y_indices = y_indices
        self.x_indices = x_indices

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return {
            "patch": self.patches[index],
            "y_index": self.y_indices[index],
            "x_index": self.x_indices[index]
        }

if __name__ == "__main__":
    # input the experiment name, testing region and whether generate visual results
    exp_name = "lsr"
    visual = False
    state = "ny_1m_2013"

    # make sure data are put into following dir
    test = Test(
        "./chesapeake_data",
        "./chesapeake_data/{}_extended-test_tiles.csv".format(state),
        "./exp_{}/test/".format(exp_name),
        "exp_{}/checkpoints/best_model.pth".format(exp_name),
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        n_classes=4,
        input_size=256,
    )

    test.run_on_tiles(get_color=visual)