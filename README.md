# lsr_pytorch

This is an PyTorch implementaton of label super-resolution networks.

Original paper:

Malkin, Kolya, et al. "[Label super-resolution networks.](https://openreview.net/pdf?id=rkxwShA9Ym)"  *International Conference on Learning Representations* . 2018.

Robinson, Caleb, et al. "[Large scale high-resolution land cover mapping with multi-resolution data.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Robinson_Large_Scale_High-Resolution_Land_Cover_Mapping_With_Multi-Resolution_Data_CVPR_2019_paper.pdf)"  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* . 2019.

## Date preparation

### Install azcopy

All datasets are downloaded via azcopy, so you need to install it:

```
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
```

Only works with `sudo azcopy`.

### Download data

Run `download_all.sh` script to download all data via azcopy. The script will create a folder name `chesapeake_data` in the project root.

The downloaded dataset contains raw images, visual class maps, and preprocessed `.npy` patches that integrate all data including high-resolution imagery, high-resolution labels, low-resolution NLCD labels.

The script will download data in all regions by default. You could change the script to download only some of them easily.

## Requirements

You need to confirm you have these libraries before running codes:

- numpy
- pandas
- PyTorch
- segmentation_models_pytorch
- Albumentations
- rasterio
- wandb
- tqdm

## Training

By using `train.py`, you could train models using the chesapeake dataset.

Please make sure in the codes that you have specified the right path to chesapeake_data and the states of data you want to train on:

```python
# config the path to the chesapeake dataset folder and states trained on
data_dir = "chesapeake_data/"
state = ["ny_1m_2013"]
```

Then run `train.py` by providing experiment setups:

```Python
parser = argparse.ArgumentParser(description='Train the UNet for land cover mapping')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=20, help='Batch size')
parser.add_argument('--size', dest='image_size', type=int, default=256, help="Image size of input patches. If smaller than original size, random crop will be done.")
parser.add_argument('--step', type=int, default=300, help="Number of iterations for every epoch")
parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
 parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
parser.add_argument('--encoder', default="se_resnext50_32x4d", type = str, help="Encoder type for neural network")
parser.add_argument('--weights', default=None, type = str, help="Pretrained weights for models")
parser.add_argument("--lr-scheduler", dest="lr_scheduler", action='store_true', default=False, help="Scheduler for learning rate")
parser.add_argument('--scheduler-step', dest='scheduler_step', type=int, default=5, help="Number of epochs before learning rate change")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument('--name', type = str, help = 'name of experiment', required = True)
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
parser.add_argument('--superres', action='store_true', default=False, help="Use super-resolution loss")
parser.add_argument('--naive', action='store_true', default=False, help="Use soft naive loss")
parser.add_argument('--seed', type=int, default=0)
```

Tips:

- Provide the experiment name via `--name`. A fold named `exp_${name}` will be created to contain the information about this experiment, including a log and the best model.
- This script could do experiments using three methods: hr - trained with ground-truth high-resolution labels, superres - using label super-resolution loss, naive - using soft naive loss. Use arguments `--superres` or `--naive` to declare these two methods respectively. If not any of them is provided, it will run hr experiment by default.

## Testing

Run `testing.py` to evaluate pre-trained model on testing set and generate visual prediction maps.

You can assign the experiment name of the model you want to test. And also control the testing region and whether to generate visual results:

```python
# input the experiment name and whether generate visual results
exp_name = "lsr"
visual = False
state = "ny_1m_2013"
```

By default, the script will use the best model under this exeperiment.

Also, make sure that whether the testing script use the same model architecutre in load_model function:

```python
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
```

If you generate visual prediction maps, the visual results would be stored in the folder `test/exp_${name}`.
