import argparse
import logging
import sys
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
from loss import SRLoss
from evaluate import multiclass_jaccard, evaluate, jaccard
from dataset import LandCoverDataset
from util import load_nlcd_stats

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              image_size: int = 256,
              step: int = 300,
              learning_rate: float = 0.001,
              save_checkpoint: bool = True,
              do_superres: bool = False,
              amp: bool = False):

    # config the path to the chesapeake dataset folder
    data_dir = "/home/liangcw/{}_data/".format(args.dataset)
    state = ["ny_1m_2013"]

    # 1. Create dataset
    train_set = LandCoverDataset(data_dir, (image_size, image_size, 4), state, "train", do_superres = do_superres, pre=args.pre)
    val_set = LandCoverDataset(data_dir, (image_size, image_size, 4), state, "val", do_superres = do_superres)
    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='Land cover mapping', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint, amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Scheduler:       {args.lr_scheduler}
        Scheduler step:  {args.scheduler_step}
        Training size:   {n_train}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        Super resolution:{do_superres}
        GPU:             {args.gpu}
        Encoder:         {args.encoder}
        Weights:         {args.weights}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.scheduler_step, gamma = 0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # if do super resolution
    if do_superres:
        nlcd_class_weights, nlcd_means, nlcd_vars = load_nlcd_stats(tensor=True)
        nlcd_class_weights = nlcd_class_weights.to(device=device)
        nlcd_means = nlcd_means.to(device=device)
        nlcd_vars = nlcd_vars.to(device=device)
        if n_classes == 4:
            nlcd_means = nlcd_means[:,1:]
            nlcd_vars = nlcd_vars[:,1:]
        criterion = SRLoss(nlcd_class_weights, nlcd_means, nlcd_vars)
    # if not, just use cross entropy loss
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    global_step = 0
    best_score = 0

    # 4. Begin training
    for epoch in range(epochs):
        net.train()

        iou_score = 0
        iou_class_score = 0
        epoch_loss = 0
        epoch_step = 0

        assert step * batch_size <= n_train
        epoch_train = step * batch_size

        with tqdm(total=epoch_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                labels = batch['label_hr']

                assert images.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)

                # ignore zero index
                if n_classes >= 4:
                    labels -= 1

                # if do super resolution, get nlcd label
                if do_superres:
                    y_nlcd = batch["label_nlcd"]
                    y_nlcd = y_nlcd.to(device=device, dtype=torch.long)
                    y_nlcd = F.one_hot(y_nlcd, 22).permute((0, 3, 1, 2)) # to one-hot

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    if do_superres:
                        # compute super resolution loss
                        loss = criterion(masks_pred, y_nlcd) * (1/40)
                    else:
                        # compute cross entropy loss
                        loss = criterion(masks_pred, labels)

                with torch.no_grad():
                    pred = F.softmax(masks_pred, dim = 1).argmax(dim=1)
                    # compute mIoU on training set
                    iou, iou_class = multiclass_jaccard(F.one_hot(pred, n_classes).permute((0, 3, 1, 2)).float(), F.one_hot(labels, n_classes).permute((0, 3, 1, 2)).float(), class_score=True)
                
                # gradient descent
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])

                global_step += 1
                epoch_step += 1

                epoch_loss += loss.item()
                iou_score += iou.item()
                iou_class_score += iou_class

                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'train loss': loss.item(),
                    'train mIoU': iou.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if epoch_step % step == 0:
                    break

        # Evaluation round
        val_score, val_score_class = evaluate(net, val_loader, device, do_superres=do_superres, n_classes=n_classes, dataset=args.dataset)
        experiment.log({
            'validation Score': val_score,
            'images': wandb.Image(images[0].cpu()),
            'masks': {
                'true': wandb.Image(labels[0].float().cpu()),
                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
        })

        if args.lr_scheduler:
            scheduler.step(val_score)

        # record logs
        logging.info(f"********** Epoch {epoch + 1} **********")
        logging.info('Training average loss: {}, Training mIoU: {}, Training class IoU: {}'.format(epoch_loss / epoch_step, iou_score / epoch_step, iou_class_score / epoch_step))
        logging.info('Validation IoU: {}, Validation class IoU: {}'.format(val_score, val_score_class))
        # save the best model according to mIoU on validation set
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            current_score = val_score
            if current_score > best_score:
                torch.save(net.state_dict(), str(dir_checkpoint / 'best_model.pth'))
                best_score = current_score
                logging.info(f'Best model saved!')
        logging.info(f"**********************************************")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=20, help='Batch size')
    parser.add_argument('--size', dest='image_size', type=int, default=256)
    parser.add_argument('--step', type=int, default=300)
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--encoder', default="se_resnext50_32x4d", type = str)
    parser.add_argument('--weights', default=None, type = str)
    parser.add_argument("--lr-scheduler", dest="lr_scheduler", action='store_true', default=False)
    parser.add_argument('--scheduler-step', dest='scheduler_step', type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--name', type = str, help = 'name of experiment', required = True)
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--superres', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set experiment
    dir_exp = "exp_{}/".format(args.name)
    Path(dir_exp).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename = os.path.join(dir_exp, "run.log"), filemode='w', level=logging.INFO, format='%(levelname)s: %(message)s')
    dir_checkpoint = Path(os.path.join(dir_exp, 'checkpoints/'))

    # set device: CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    n_channels = 4
    n_classes = 4
    save_checkpoint = True

    # use segmentation model
    net = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=args.weights,
        decoder_use_batchnorm=True,
        decoder_attention_type=scse,
        in_channels=n_channels,
        classes=n_classes,
    )

    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n')
    
    # load model
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  device=device,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  image_size=args.image_size,
                  step=args.step,
                  learning_rate=args.lr,
                  save_checkpoint=save_checkpoint,
                  do_superres=args.superres,
                  amp=args.amp,
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)