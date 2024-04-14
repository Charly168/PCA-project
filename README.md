# PCA-project
This repository shows some interesting experiments about principle component analysis(PCA) done by myself

## First Experiments
I aim to determine the principle axis of  ![example1](./finding_principle_axis/example.png), 
, which is a 3D medical image. However, I lack an effective method to calculate its principal axis directly. Hence, I employ PCA to identify its principal axis.

The approach is as follows: treating each pixel (voxel) as a sample. For instance, if the image size is 240x240x240, then I have $240^3$ samples, with the coordinate information serving as features. The purpose of PCA is to reduce feature dimensionality. Given that the image has three orthogonal axes, if I can identify a principal eigenvector and align along its direction, it would represent the principal axis.

Below are the results, where I manually segment each part and identify its principal axis.
![result1](./finding_principle_axis/result.png)