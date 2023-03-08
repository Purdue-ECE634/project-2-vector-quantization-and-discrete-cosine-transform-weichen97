import cv2
import numpy as np
import glob
import os 

def img_to_vec(img_list, block_size=4):
    img_vec = []
    for img in img_list:
        # make sure the image size is divisible by block size
        if img.shape[0] % block_size != 0 or img.shape[1] % block_size != 0:
            new_h = np.floor(img.shape[0] // block_size) * block_size
            new_w = np.floor(img.shape[1] // block_size) * block_size
            img = cv2.resize(img, (int(new_w), int(new_h)))
        assert img.shape[0] % block_size == 0 and img.shape[1] % block_size == 0

        # extract image block and convert to vector
        for i in range(0, img.shape[0], block_size):
            for j in range(0, img.shape[1], block_size):
                block = img[i:i + block_size, j:j + block_size]
                img_vec.append(block)

    return np.array(img_vec)

def MSE(img1, img2):
    return np.mean((img1 - img2) ** 2)

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return np.inf
    else:
        return 20 * np.log10(255.0 / np.sqrt(mse))

def vector_quantization(image, train_img_list, block_size=4, quant_level=128):
    train_list = img_to_vec(train_img_list, block_size)
    len_train = train_list.shape[0]

    # initialize codebook
    codebook = np.arange(0, 256, 256 // quant_level)
    codebook = np.tile(codebook.reshape((quant_level, 1)), 16)
    codebook = codebook.reshape((quant_level, 4, 4))
    # print(codebook.shape)
    prev_error = 1
    T = 0.01

    # train codebook
    while True:
        error = np.zeros((len_train))
        code = np.zeros((len_train))

        for i in range(len_train):
            min_mse = np.inf
            min_idx = 0

            for j in range(quant_level):
                cur_mse = MSE(train_list[i], codebook[j])
                
                if cur_mse < min_mse:
                    min_mse = cur_mse 
                    min_idx = j
            error[i] = min_mse
            code[i] = min_idx

        cur_erorr = np.mean(error)
        if (np.abs(cur_erorr - prev_error) / prev_error) <= T:
            break

        else:
            for l in range(quant_level):
                if np.sum(code == l) != 0: 
                    codebook[l] = np.mean(train_list[code == l], axis=0)
                else:
                    codebook[l] = np.zeros((4, 4))
                    prev_error = cur_erorr        
               
    # quantize image with trained codebook
    quantized_image = np.zeros_like(image)
    for i in range(image.shape[0] // block_size):
        for j in range(image.shape[1] // block_size):
            min_mse = np.inf
            index = 0
            for k in range(quant_level):
                img_block = image[4*i:4*(i+1), 4*j:4*(j+1)]
                cur_mse = MSE(img_block, codebook[k])

                if cur_mse < min_mse:
                    min_mse = cur_mse
                    index = k
            
            quantized_image[4*i: 4*(i+1), 4*j: 4*(j+1)] = codebook[index]

    return quantized_image

if __name__=="__main__":
    img_path = '../sample_image/'
    img = cv2.imread(img_path+'fruits.png', cv2.IMREAD_GRAYSCALE)
    
    train_img_path = glob.glob(img_path + '*.png')[:10]
    train_img_path.sort()
    train_img_list = []
    for name in train_img_path:
        train_img_list.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))
    
    # single image training
    for L in [16, 32, 64, 128]:
        quantized_img = vector_quantization(img, [img], block_size=4, quant_level=L)
        cv2.imwrite('quantized_image_trained_by_1_L_%d.png'%L, quantized_img)
        psnr = PSNR(quantized_img, img)
        print('Finished writing 1-image reconstruction with level=%d'%L)
        print('L=%d, PSNR=%.2f'%(L, psnr))

    # 10 images training
    for L in [16, 32, 64, 128]:
        quantized_img = vector_quantization(img, train_img_list, block_size=4, quant_level=L)
        cv2.imwrite('quantized_image_trained_by_10_L_%d.png'%L, quantized_img)
        psnr = PSNR(quantized_img, img)
        print('Finished writing 10-image reconstruction with level=%d'%L)
        print('L=%d, PSNR=%.2f'%(L, psnr))