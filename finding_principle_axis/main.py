import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import SimpleITK as sitk
import os


def points_metric(im,value):
    points = []
    depth,height,width = im.shape
    for i in range(depth):
        for j in range(height):
            for k in range(width):
                if im[i, j,k] == value:
                    points.append([float(i),float(j), float(k)])
    return points

def label_size(label_np,value):

    size = coronal_start = 0
    segmentation = np.where(label_np == value)

    try:
        if len(segmentation[0]) != 0 and len(segmentation[1]) != 0 and len(segmentation[2] != 0):
            c_min = int(np.min(segmentation[0]))
            c_max = int(np.max(segmentation[0]))
            y_min = int(np.min(segmentation[1]))
            y_max = int(np.max(segmentation[1]))
            x_min = int(np.min(segmentation[2]))
            x_max = int(np.max(segmentation[2]))

            depth = c_max - c_min
            height = y_max - y_min
            width = x_max - x_min
            size = (depth,height,width)
            coronal_start = y_max
    except ValueError as e:
        print("trobule with {label_file}")
        print(str(e))


    return size,coronal_start



def pca(X, k=1):  # Reduce to k dimensions
    mean = np.mean(X, axis=1)  # axis=0 calculates mean along columns, axis=1 calculates mean along rows
    X = X - mean
    # Calculate covariance between different dimensions, not between samples, method 1:
    C = np.dot(X, X.T)
    e, EV = np.linalg.eig(np.mat(C))  # Compute eigenvalues and eigenvectors of the covariance matrix, e is eigenvalues, EV is eigenvectors
    e_idx = np.argsort(-e)[:k]  # Get indices of the top k largest eigenvalues
    EV_main = EV[:, e_idx]  # Get eigenvectors corresponding to the eigenvalues (indices), as main components
    low = np.dot(EV_main.T, X)  # This is the projection of the original dataset onto the main components
    return low, EV_main, mean



def scatter_vector(ax,im, value, color):

    points4 = points_metric(im, value)
    im = np.mat(points4).T
    low, EV_main, mean = pca(im)
    # recon = np.dot(EV_main, low) + mean
    # margin = im.shape[1] - mean[1]
    ax.scatter(im[0].A[0], im[1].A[0], im[2].A[0], s=1,c=color, alpha=0.5)
    #
    # start_point_3 = np.array([[float(mean_3[0]),float(margin-30),float(mean_3[2])]]) # Mean point as the starting point
    start_point = mean.flatten()  # Mean point as the starting point
    for i in range(EV_main.shape[1]):  # For each eigenvector
        end_point = start_point + EV_main[:, i].T * 2000  # End point of the arrow
        ax.quiver(start_point[0, 0], start_point[0, 1], start_point[0, 2],
                  end_point[0, 0], end_point[0, 1], end_point[0, 2],
                  color='r', length=0.1, arrow_length_ratio=0.3)

    return EV_main, start_point


def angle_cal(v1,v2):
    dot_product = np.dot(v1.T,v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm1 * norm2)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)

def norm_vector(center,v):
    new_vector = vector_through_center(center,v)
    return v / np.linalg.norm(new_vector)

def vector_through_center(center,v):
    return center + v

def euclidean_distance(v1,v2):
    return np.linalg.norm(v1 - v2)


def main(path):
    img_sitk = sitk.ReadImage(path)
    im = sitk.GetArrayFromImage(img_sitk)
    shape = im.shape
    new_im = np.zeros_like(im)

    M3_size,start = label_size(im,1)

    bone_size,start1 = label_size(im,2)

    if start + 10 > start1 and start < start1:
        bone_start = start - 10
        bone_end = start1
    elif start + M3_size[2] < start1 and start < start1:
        bone_start = start + 5
        bone_end = start+M3_size[2]
    elif start + M3_size[2] > start1 and start < start1:
        bone_start = start + 5
        bone_end = start1
    else:
        bone_start = start - 10 #  start > start1
        bone_end = start
    # new_im = im[:,(M3_size[2]):,:]
    print(M3_size[2])
    new_im[:,bone_start:bone_end,:] = im[:,bone_start:bone_end,:]

    new_im2 = np.zeros_like(im)
    new_im2[:,0:M3_size[2],:] = im[:,0:M3_size[2],:]
    # new_im[:,0:M3_size[2],:] = im[:,0:M3_size[2],:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    EV_main1,start_point1 = scatter_vector(ax,im,1,"blue") # M3
    EV_main2,start_point2 = scatter_vector(ax,im,2,"red")  # bone
    EV_main3,start_point3 = scatter_vector(ax,new_im,2,"green")  #M2
    EV_main4,start_point4 = scatter_vector(ax,new_im2,2,"orange") # cheekbone

    angle_M3_bone = angle_cal(EV_main1,EV_main2)
    angle_M3_M2 = angle_cal(EV_main1,EV_main3)

    if angle_M3_M2 < 15:
        new_im2[:, 0:int(M3_size[2] / 2), :] = im[:, 0:int(M3_size[2] / 2), :]
        EV_main4,start_point4 = scatter_vector(ax,new_im2,2,"orange") # cheekbone

    angle_M3_cheekbone = angle_cal(EV_main1,EV_main4)

    v1 = norm_vector(start_point1,EV_main1)
    v2 = norm_vector(start_point3,EV_main3)
    euclidean_distance_M2_M3 = euclidean_distance(v1,v2)



    print("EV_main of M3: {}\n".format(EV_main1) +
          "EV_main2 of bone: {}\n".format(EV_main2) +
          "EV_main3 of M2: {}\n".format(EV_main3) +
          "EV_main4 of Cheekbone: {}\n".format(EV_main4))

    print("start_point1 of M3: {}\n".format(start_point1) +
          "start_point2 of bone: {}\n".format(start_point2) +
          "start_point3 of M2: {}\n".format(start_point3) +
          "start_point4 of cheekbone: {}\n".format(start_point4))

    print("angle between M3 and bone: {}\n".format(angle_M3_bone) +
          "angle between M3 and M2: {}\n".format(angle_M3_M2) +
          "angle between M3 and Cheekbone: {}\n".format(angle_M3_cheekbone) +
          "euclidean_distance between M3 and M2: {}\n".format(euclidean_distance_M2_M3))

    plt.show()
    return angle_M3_bone,angle_M3_M2,angle_M3_cheekbone,euclidean_distance_M2_M3


if __name__ == "__main__":
    label = "/home/charly/Downloads/disk/mandibular_model/ensemble_model/pca/demo/0/004/0_004_label.nrrd"

    angle_M3_bone,angle_M3_M2,angle_M3_cheekbone,euclidean_distance_M2_M3 = main(label)
    print("*******************************************")


    a = 1
