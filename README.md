# lsr_pytorch

This is an PyTorch implementaton of label super-resolution networks.

Original paper: 

Malkin, Kolya, et al. "Label super-resolution networks."  *International Conference on Learning Representations* . 2018. https://openreview.net/pdf?id=rkxwShA9Ym

Robinson, Caleb, et al. "Large scale high-resolution land cover mapping with multi-resolution data."  *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* . 2019.

## Date preparation

### Install azcopy

The datasets are downloaded via azcopy

```
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
```

Only works with `sudo azcopy`.

### Download data
