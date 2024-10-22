# DCMamba

### Introduction

This is the Implementation of《Diversity-enhanced Collaborative Mamba for Semi-supervised Medical Image Segmentation》

1. Prepare the dataset
Download Synapse for Semi-/Supervised learning through [[Google Drive]](https://drive.google.com/file/d/1pO_YBx_3OCzadYQXzKsUqmXH6Ghv-z2y/view?usp=sharing)

3. Prepare the pre-trained weights
Download through [[Google Drive]](https://drive.google.com/file/d/1uUPsr7XeqayCxlspqBHbg5zIWx0JYtSX/view?usp=sharing) for Mamba-UNet, and save in `../code/pre_trained_weights`.

4. Train the model
```
python train_pln.py --exp model_name
```

### Acknowledgement

Part of the code is origin from [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet/tree/main). 
