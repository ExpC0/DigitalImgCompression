import cv2
import numpy as np
import math
from zigzag import zigzag

def golomb_encode(value, m):
    value = int(value)  # Ensure the value is an integer
    q = value // m
    r = value % m
    quotient = '1' * q + '0'
    remainder = '{:0{width}b}'.format(r, width=int(math.log2(m)))
    return quotient + remainder

def get_golomb_encoding(arranged):
    bitstream = ""
    for value in arranged:
        bitstream += golomb_encode(value, 2)  # Golomb parameter m=2
    return bitstream

# 2D Discrete Cosine Transform (DCT)
def dct_2d(block):
    m, n = block.shape
    dct = np.zeros((m, n))
    for u in range(m):
        for v in range(n):
            sum = 0
            for i in range(m):
                for j in range(n):
                    sum += block[i, j] * math.cos((2 * i + 1) * u * math.pi / (2 * m)) * math.cos((2 * j + 1) * v * math.pi / (2 * n))
            cu = 1 / math.sqrt(2) if u == 0 else 1
            cv = 1 / math.sqrt(2) if v == 0 else 1
            dct[u, v] = 0.25 * cu * cv * sum
    return dct

# defining block size
block_size = 8

# Quantization Matrix
# we can quantization matrix for lower quality (higher compression)

QUANTIZATION_MAT = np.array([
    [80, 60, 50, 80, 120, 180, 220, 250],
    [55, 60, 70, 95, 130, 255, 255, 255],
    [70, 65, 80, 120, 200, 255, 255, 255],
    [70, 85, 110, 145, 255, 255, 255, 255],
    [90, 110, 185, 255, 255, 255, 255, 255],
    [120, 175, 255, 255, 255, 255, 255, 255],
    [245, 255, 255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255, 255, 255]
])


# Function to compress a single channel
def compress_channel(channel):
    # get size of the channel
    [h, w] = channel.shape

    # No of blocks needed : Calculation
    height = h
    width = w
    h = np.float32(h)
    w = np.float32(w)

    nbh = math.ceil(h / block_size)
    nbh = np.int32(nbh)

    nbw = math.ceil(w / block_size)
    nbw = np.int32(nbw)

    # Pad the image, because sometimes image size is not divisible by block size
    # get the size of padded image by multiplying block size by number of blocks in height/width

    # height of padded image
    H = block_size * nbh

    # width of padded image
    W = block_size * nbw

    # create a numpy zero matrix with size of H, W
    padded_img = np.zeros((H, W))

    # copy the values of channel into padded_img[0:h, 0:w]
    padded_img[0:height, 0:width] = channel[0:height, 0:width]

    for i in range(nbh):
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            block = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2]


            # apply 2D discrete cosine transform to the selected block
            DCT = dct_2d(block)


            DCT_normalized = np.divide(DCT, QUANTIZATION_MAT).astype(int)

            # reorder DCT coefficients in zigzag order by calling zigzag function
            # it will give a one-dimensional array



            reordered = zigzag(DCT_normalized)

            # reshape the reordered array back to (block size by block size)
            reshaped = np.reshape(reordered, (block_size, block_size))

            # copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = reshaped

    return padded_img

img = cv2.imread('Lena.jpg')

# Splitting the image into R, G, B channels
B, G, R = cv2.split(img)

print("applying inverse discrete cosine transform...")
print("quantization...")
print("applying zigzag scanning...")

# encoding each channel
encoded_B = compress_channel(B)
encoded_G = compress_channel(G)
encoded_R = compress_channel(R)

# Saving the encoded channels
cv2.imwrite('output_files/B_channel.jpg', np.uint8(encoded_B))
cv2.imwrite('output_files/G_channel.jpg', np.uint8(encoded_G))
cv2.imwrite('output_files/R_channel.jpg', np.uint8(encoded_R))

cv2.imshow('applying dct,zigzag,quantization in R channel',encoded_R)
cv2.waitKey(0)

cv2.imshow('applying dct,zigzag,quantization in G channel',encoded_G)
cv2.waitKey(0)

cv2.imshow('applying dct,zigzag,quantization in B channel',encoded_B)

cv2.waitKey(0)

# Arranging and encoding
arranged_B = encoded_B.flatten()
arranged_G = encoded_G.flatten()
arranged_R = encoded_R.flatten()

arranged_B = arranged_B.astype(int)
arranged_G = arranged_G.astype(int)
arranged_R = arranged_R.astype(int)

# Golomb encoded data is written to separate text files
bitstream_B = get_golomb_encoding(arranged_B)
bitstream_G = get_golomb_encoding(arranged_G)
bitstream_R = get_golomb_encoding(arranged_R)

# Adding size information and semicolon to denote end of image
bitstream_B = f"{encoded_B.shape[0]} {encoded_B.shape[1]} {bitstream_B};"
bitstream_G = f"{encoded_G.shape[0]} {encoded_G.shape[1]} {bitstream_G};"
bitstream_R = f"{encoded_R.shape[0]} {encoded_R.shape[1]} {bitstream_R};"

# Written to separate text files
with open("image_B.txt", "w") as file1:
    file1.write(bitstream_B)

with open("image_G.txt", "w") as file2:
    file2.write(bitstream_G)

with open("image_R.txt", "w") as file3:
    file3.write(bitstream_R)

print("Golumb encoded values are saved in image_B.txt,image_G.txt,image_R.txt")


cv2.destroyAllWindows()
