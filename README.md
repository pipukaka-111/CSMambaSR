## CSMambaSR: Cross Scale Mamba for Remote Sensing Image Super-Resolution

> **Abstract:**  The Mamba model shows potential in super-resolution reconstruction of remote sensing images by virtue of the global receptive field advantage of linear computational complexity. However, current Mamba-based RSISR methods are mostly confined to single-scale feature extraction, limiting performance improvement. This paper proposes a Cross-scale Mamba Super-Resolution Network (CSMambaSR). It incorporates the Cross-scale 2D-Selective Scanning Module (CSSS2D) and the Cross-scale Visual State Space Module (CSVSSM), enabling cross-attention operations with a global receptive field to be accomplished with linear computational complexity based on the Mamba framework. Additionally, the Cross-scale Residual State-Space Block (CSRSSB) is designed on this basis, allowing the network to simultaneously process and fuse feature maps of two scales. Additionally, CSMambaSR innovatively introduces the Channel Attention Back-Projection Reconstruction (CABR), which fuses information from the original low-resolution input during the reconstruction phase, further enhancing the quality of image super-resolution reconstruction. Experiments on the UCMerced and AID datasets demonstrate that, compared to other mainstream methods, CSMambaSR significantly enhances the performance of the RSISR task by strengthening its cross-scale information mining and fusion capabilities.



## <a name="installation"></a> :wrench: Installation

This codebase was tested with the following environment configurations. It may work with other versions.

- Ubuntu 20.04
- CUDA 11.7
- Python 3.9
- PyTorch 2.0.1 + cu117

### Previous installation
To use the selective scan with efficient hard-ware design, the `mamba_ssm` library is needed to install with the folllowing command.

```
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

## <a name="training"></a> Training

### Train

```
python basicsr/train.py -opt options/train/mambairca_UCMerced_train_bicubic_x2.yml
```


## <a name="testing"></a> Testing

### Test

```
python basicsr/test.py -opt options/test/mambair_UCMerced_test_bicubic_x2.yml
```

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [MambaIR](https://github.com/csguoh/MambaIR), Thanks for their awesome work.
