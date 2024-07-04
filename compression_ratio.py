import os

def calculate_compression_ratio(original_file, compressed_file):

    original_size = os.path.getsize(original_file)
    compressed_size = os.path.getsize(compressed_file)

    if compressed_size == 0:
        raise ValueError("Compressed file size is zero")

    compression_ratio = original_size / compressed_size
    return compression_ratio



original_size = os.path.getsize('Lena.jpg')
compressed_size = os.path.getsize('decoded_rgb_image.jpg')

print(f"Original Size: {original_size} bytes")
print(f"Compressed Size: {compressed_size} bytes")


original_file = 'Lena.jpg'
compressed_file = 'decoded_rgb_image.jpg'

compression_ratio = calculate_compression_ratio(original_file, compressed_file)
print(f"Compression Ratio: {compression_ratio:.2f}")