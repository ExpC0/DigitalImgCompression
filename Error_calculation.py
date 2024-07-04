import cv2
import numpy as np


def calculate_difference(original_img_path, reconstructed_img_path):

    original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    reconstructed_img = cv2.imread(reconstructed_img_path, cv2.IMREAD_COLOR)

    # Ensure both images are of the same size
    if original_img.shape != reconstructed_img.shape:
        raise ValueError("Original image and reconstructed image must have the same dimensions.")

    # Calculate the absolute difference between the images
    difference = cv2.absdiff(original_img, reconstructed_img)

    # Calculate statistics
    mse = np.mean((original_img - reconstructed_img) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')

    # Save and display the difference image
    cv2.imwrite('output_files/difference_image.jpg', difference)
    cv2.imshow('Difference Image', difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mse, psnr



original_img_path = 'Lena.jpg'
reconstructed_img_path = 'decoded_rgb_image.jpg'

mse, psnr = calculate_difference(original_img_path, reconstructed_img_path)
print(f'Mean Squared Error (MSE): {mse}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr} dB')
