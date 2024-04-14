import numpy as np
import SimpleITK as sitk
import cv2

path = "/home/charly/Downloads/disk/mandibular_model/ensemble_model/pca/Lenna.jpg"
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) # (316x316  X  1)
cv2.imshow("gray_img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 316x316 >>> 76x76x4x4  (4x4 ---- 2x2)
# print(img.shape)
# X = img.flatten().reshape(1,-1)
# # X = np.array(X)
# # X = X.reshape(1,-1)# (1 x 316`2)
# print(X.shape)
# mean = np.mean(X,axis=1)
# X_mean = X - mean
# cov_matrix = np.cov(X_mean.T) # (316`2 x 316`2)
# ein_values,ein_vector = np.linalg.eig(cov_matrix)
# top_vec100 = ein_vector[:100] # (316`2 x 100)
# reconstruct = np.dot(top_vec100,X_mean) + mean
# print(reconstruct.shape)
# new_img = reconstruct.reshape(img.shape)
# print(new_img.shape)
#
# cv2.imshow("rec_img",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

block_size = 4
num_blocks = img.shape[0] // block_size  # 块的数量
blocks = []
for i in range(num_blocks):
    for j in range(num_blocks):
        block = img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        block = block.flatten().reshape(1,-1)
        blocks.append(block)
blocks = np.array(blocks)
print(blocks.shape)
blocks = blocks.reshape(-1,16)
block_reshape = blocks.reshape(-1,16)
print(blocks.shape)

mean = np.mean(blocks,axis=1, keepdims=True)
X_mean = blocks - mean # (79`2 x 16)
conv = np.cov(X_mean.T) # (16 x 79`2)
ein_values,ein_vectors = np.linalg.eig(conv)
ein_vectors_top10 = ein_vectors[:,:4] # (16x4)
recon_X = np.dot(ein_vectors_top10.T,X_mean.T).T + mean
print(recon_X.shape)

reconstructed_blocks = recon_X.reshape(num_blocks, num_blocks, 2, 2)
reconstructed_img = np.block([[reconstructed_blocks[i, j] for j in range(num_blocks)] for i in range(num_blocks)])
reconstructed_img = reconstructed_img.astype(np.uint8)
# reconstructed_img = cv2.resize(reconstructed_img,img.shape)

cv2.imshow("reconstructed_img", reconstructed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




