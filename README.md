# DCMamba

### Introduction

This is the Implementation of《Diversity-enhanced Collaborative Mamba for Semi-supervised Medical Image Segmentation》

### Usage
1. Clone the repository:
```
git clone https://github.com/ShumengLI/DCMamba.git 
cd DCMamba
```

2. Prepare the dataset

Download Synapse dataset through [[Google Drive]](https://drive.google.com/file/d/1pO_YBx_3OCzadYQXzKsUqmXH6Ghv-z2y/view?usp=sharing)

3. Prepare the pre-trained weights

Download through [[Google Drive]](https://drive.google.com/file/d/1uUPsr7XeqayCxlspqBHbg5zIWx0JYtSX/view?usp=sharing) for Mamba-UNet, and save in `./pre_trained_weights`.

4. Train the model
```
python train_dcmamba.py --exp dcmamba --dataset synapse --num_classes 9 --labeled_num 4
```

### Acknowledgement

Part of the code is origin from [VMamba](https://github.com/MzeroMiko/VMamba) and [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet/tree/main). 
