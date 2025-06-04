from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Step 1: Load image and print dimensions ---
image_path = "baby.jpg"
img = Image.open(image_path)
width, height = img.size
print("Image dimensions:", width, "x", height)

# RGB 24-bit raw size
uncompressed_size = width * height * 3  # bytes
file_size = os.path.getsize(image_path)  # bytes
compression_rate = uncompressed_size / file_size
print(f"Compression Rate: {compression_rate:.2f}")

# --- Step 2
ycbcr_img = img.convert('YCbCr')
ycbcr = np.array(ycbcr_img)
ycbcr[:, :, 0] = np.clip(ycbcr[:, :, 0] + 50, 0, 255)  # brighten Y
bright_img = Image.fromarray(ycbcr, 'YCbCr').convert('RGB')
bright_img.save("output_step2_brightened.jpg")
print("Step 2: Brightened image saved as output_step2_brightened.jpg")

# --- Step 3
ycbcr = np.array(img.convert('YCbCr'))
cr = ycbcr[:, :, 2]
mask = cr > 150
ycbcr[mask, 2] = 0
red_suppressed_img = Image.fromarray(ycbcr, 'YCbCr').convert('RGB')
red_suppressed_img.save("output_step3_red_removed.jpg")
print("Step 3: Red suppressed image saved as output_step3_red_removed.jpg")

# --- Step 4

def down_up_sample(comp):
    comp_down = comp[::2, ::2]
    comp_up = np.repeat(np.repeat(comp_down, 2, axis=0), 2, axis=1)
    return comp_up[:height, :width]

ycbcr = np.array(img.convert('YCbCr'))
Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
Cb_recon = down_up_sample(Cb)
Cr_recon = down_up_sample(Cr)
jpeg_like_ycbcr = np.stack((Y, Cb_recon, Cr_recon), axis=2).astype(np.uint8)
jpeg_like_img = Image.fromarray(jpeg_like_ycbcr, 'YCbCr').convert('RGB')
jpeg_like_img.save("output_step4_jpeg_like.jpg")
print("Step 4: JPEG-like chroma subsampled image saved as output_step4_jpeg_like.jpg")

# --- Step 5
Y_recon = down_up_sample(Y)
full_down_ycbcr = np.stack((Y_recon, Cb_recon, Cr_recon), axis=2).astype(np.uint8)
full_down_img = Image.fromarray(full_down_ycbcr, 'YCbCr').convert('RGB')
full_down_img.save("output_step5_full_downsampled.jpg")
print("Step 5: Fully downsampled image saved as output_step5_full_downsampled.jpg")


def show_images_side_by_side(titles, images):
    plt.figure(figsize=(15, 5))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_images_side_by_side(
    ["Original", "Brightened", "Red Removed"],
    [img, bright_img, red_suppressed_img]
)

show_images_side_by_side(
    ["JPEG-like (Cb, Cr)", "Full Downsampled (Y, Cb, Cr)"],
    [jpeg_like_img, full_down_img]
)
