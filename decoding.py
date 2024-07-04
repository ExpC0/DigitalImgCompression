import numpy as np
import cv2
import math
from zigzag import inverse_zigzag

def golomb_decode(bitstream, m):
    index = 0
    values = []

    while index < len(bitstream):
        # Find the length of the quotient
        quotient_len = 0
        while bitstream[index] == '1':
            quotient_len += 1
            index += 1
        index += 1  # skip the '0' separator

        # Read the remainder
        remainder = bitstream[index:index + int(math.log2(m))]
        remainder = int(remainder, 2)
        index += int(math.log2(m))

        value = quotient_len * m + remainder
        values.append(value)

    return values

# Define the 2D Inverse Discrete Cosine Transform (IDCT)
def idct_2d(block):
    m, n = block.shape
    idct = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            sum = 0
            for u in range(m):
                for v in range(n):
                    cu = 1 / math.sqrt(2) if u == 0 else 1
                    cv = 1 / math.sqrt(2) if v == 0 else 1
                    sum += cu * cv * block[u, v] * math.cos((2 * i + 1) * u * math.pi / (2 * m)) * math.cos((2 * j + 1) * v * math.pi / (2 * n))
            idct[i, j] = 0.25 * sum
    return idct

# Quantization Matrix
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

def decompress_channel(bitstream):
    # Extract the dimensions of the padded image
    dimensions, bitstream = bitstream.split(" ", 2)[0:2], bitstream.split(" ", 2)[2]
    H, W = int(dimensions[0]), int(dimensions[1])

    # Decode the Golomb encoded bitstream
    decoded_values = golomb_decode(bitstream[:-1], 2)  # Remove the semicolon

    # Reshape the decoded values back to the padded image
    padded_img = np.array(decoded_values).reshape(H, W)

    # Inverse quantization and IDCT for each block
    block_size = 8
    nbh = H // block_size
    nbw = W // block_size

    for i in range(nbh):
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            block = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2]

            # Inverse zigzag order
            rearranged = inverse_zigzag(block.flatten(), block_size, block_size)

            # dequantization
            dequantized = rearranged * QUANTIZATION_MAT

            # Apply IDCT
            IDCT = idct_2d(dequantized)

            # Copy the IDCT block back to the image

            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = IDCT

    # Convert to uint8
    dencoded_img = np.uint8(padded_img[0:H, 0:W])

    return dencoded_img

# Reading the compressed bitstreams
with open("image_B.txt", "r") as file_B:
    bitstream_B = file_B.read()

with open("image_G.txt", "r") as file_G:
    bitstream_G = file_G.read()

with open("image_R.txt", "r") as file_R:
    bitstream_R = file_R.read()

print("golumb coded values are loaded......")

print("applying inverse zigzag scanning...")
print("dequantization...")
print("applying inverse discrete cosine transform...")

# Decompress each channel
dencoded_B = decompress_channel(bitstream_B)
dencoded_G = decompress_channel(bitstream_G)
dencoded_R = decompress_channel(bitstream_R)

# Save the dencoded channels
cv2.imwrite('output_files/decoded_B_channel.jpg', dencoded_B)
cv2.imwrite('output_files/decoded_G_channel.jpg', dencoded_G)
cv2.imwrite('output_files/decoded_R_channel.jpg', dencoded_R)

cv2.imshow('applying inverse zigzag,dequantization, idct in R channel',dencoded_R)
cv2.waitKey(0)

cv2.imshow('applying inverse zigzag,dequantization,idct in G channel',dencoded_G)
cv2.waitKey(0)

cv2.imshow('applying inverse zigzag, dequantization, idct  in B channel',dencoded_B)

cv2.waitKey(0)

# Merge the channels to get the RGB image
dencoded_img = cv2.merge((dencoded_B, dencoded_G, dencoded_R))

# Save the merged RGB image
cv2.imwrite('decoded_rgb_image.jpg', dencoded_img)

# Display the dencoded RGB image
cv2.imshow('decoded RGB Image', dencoded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
