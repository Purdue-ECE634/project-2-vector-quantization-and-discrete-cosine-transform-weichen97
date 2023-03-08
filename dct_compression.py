import cv2
import numpy as np
from scipy.fftpack import dct, idct


def dct2d(img):
    # 2D DCT
    out = dct(dct(img, axis = 0, norm='ortho'), axis = 1, norm='ortho')
    return out 

def idct2d(img):
    # 2D IDCT
    out = idct(idct(img, axis = 0, norm='ortho'), axis = 1, norm='ortho')
    return out 

def PSNR(img1, img2):
    # calculate mean square error 
    mse = np.mean(np.abs(img1 - img2)**2)
    # calculate PSNR
    if mse == 0:
        return np.inf
    else:
        return 20 * np.log10(255.0 / np.sqrt(mse))

def zigzag(n=8):
    # build zigzag matrix for dct coefficient index 
    zigzag_matrix = np.zeros((n, n), dtype=np.int)
    x, y = 0, 0

    for i in range(n*n):
        zigzag_matrix[x][y] = i

        # upward direction
        if (x + y) % 2 == 0:
            if y == n - 1:
                x += 1
            elif x == 0:
                y += 1
            else:
                x -= 1
                y += 1

        # downward direction
        else:
            if x == n - 1:
                y += 1
            elif y == 0:
                x += 1
            else:
                x += 1
                y -= 1

    return zigzag_matrix

def compress_dct_map(img, K):
    h, w = img.shape
    partial_dct_map = np.zeros_like(img)	
    i = 0
    j = 0
    assert h * w >= K

    # obtain compressed dct conefficient from zigzag search
    matrix = zigzag(8)
    for k in range(K):  
        i, j = np.where(matrix == k)
        partial_dct_map[i, j] = img[i, j]

    return partial_dct_map


if __name__=="__main__":
    img_path = '../sample_image/'
    img = cv2.imread(img_path+'fruits.png', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    cv2.imwrite("gray.png", img)

    for K in [64, 56, 48, 40, 32, 16, 8, 4, 2]:
        img_rec = np.zeros_like(img)
        assert h % 8 == 0 and w % 8 == 0
        for i in range(h//8):
            for j in range(w//8):
                img_dct = dct2d(img[8*i: 8*(i+1), 8*j: 8*(j+1)])
                partial_dct_map = compress_dct_map(img_dct, K)
                img_rec[8*i: 8*(i+1), 8*j: 8*(j+1)] = idct2d(partial_dct_map)

        img_rec = img_rec.astype(np.uint8)
        psnr = PSNR(img, img_rec)
        print(K, psnr)
        cv2.imwrite("dct_%d.png"%K, img_rec)