import random

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def dataset(root):
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    split_ratio = 0.8
    for case in sorted(os.listdir(root)):
        case_path = os.path.join(root, case)
        face_files = os.listdir(case_path)
        random.shuffle(face_files)
        split_index = int(len(face_files) * split_ratio)
        train_imgs.extend(os.path.join(case_path, face) for face in face_files[:split_index])
        train_labels.extend([case] * split_index)

        test_imgs.extend(os.path.join(case_path, face) for face in face_files[split_index:])
        test_labels.extend([case] * (len(face_files) - split_index))
    
    return train_imgs,train_labels,test_imgs,test_labels


def Img_Embedding(img_path):
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    hight,width = img.shape
    img = img.reshape(1,hight*width)
    return img


def PCA(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_std[data_std == 0] = 1
    data_normalized = (data - data_mean) / data_std#(320,10304)
    cov_matrix = np.cov(data_normalized) #(320,200)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    selected_eigenvectors = eigenvectors[:,:40] #(320,10)

    data_projection = np.dot(data_normalized.T, selected_eigenvectors) #(10204,10)

    data_reconstruction = np.dot(data_normalized,data_projection) #(320,10)
    return data_reconstruction,data_mean,data_std,data_projection

def main(root,save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_imgpaths, train_labels, test_imgpaths, test_labels = dataset(root)

    train_imgs = [Img_Embedding(img) for img in train_imgpaths]
    train_imgs = np.concatenate(train_imgs,axis=0) #(320,10304)
    test_imgs = [Img_Embedding(img) for img in test_imgpaths]
    test_imgs = np.concatenate(test_imgs,axis=0) #(80,10304)

    train_recon, train_mean, train_std,data_projection = PCA(train_imgs) # (320,10),(10304,),(10304,)
    test_template = (test_imgs - train_mean) / train_std
    test_template = np.dot(test_template,data_projection)
    TP = 0
    ground_truth = []
    predicts = []
    for i in range(test_imgs.shape[0]):
        imgdiff = train_recon - np.tile(test_template[i,:],(train_imgs.shape[0],1))
        imgdiss = imgdiff ** 2
        sqDistances = imgdiss.sum(axis=1)
        index = np.argsort(sqDistances)[0]
        if train_labels[index] == test_labels[i]:
            TP += 1
            ground_truth.append(index)
            predicts.append(i)

    precision = TP / len(test_labels)

    # for i in range(len(ground_truth)):
    #
    #     img0 = train_imgpaths[ground_truth[i]]
    #     test0 = test_imgpaths[predicts[i]]
    #     gt = cv2.imread(img0,cv2.IMREAD_GRAYSCALE)
    #     predict = cv2.imread(test0,cv2.IMREAD_GRAYSCALE)
    #
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(gt, cmap='gray')
    #     plt.title('Ground Truth')
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(predict, cmap='gray')
    #     plt.title('Predicted')
    #     plt.savefig(os.path.join(save_path,f"{i}.png"))
    #
    #     plt.show()

    print("precision: ",precision)
if __name__ == "__main__":
    root_path = "/home/charly/Videos/PCA-project/archive"
    save_path = "/home/charly/Videos/PCA-project/face_recognition/predict"
    main(root_path,save_path)









