import os
import argparse

from PIL import Image
from tqdm import tqdm


def overlay_images_with_masks(images_folder: str,
                              masks_folder: str,
                              output_folder: str,
                              alpha: float = 0.5):
    """
    Overlays images with their corresponding mask images and saves the result.

    :param images_folder: Path to the folder containing the original images.
    :param masks_folder: Path to the folder containing the mask images.
    :param output_folder: Path to the folder where overlaid images will be saved.
    :param alpha: Transparency level for the mask overlay (0.0 to 1.0).
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    # Process each image
    for image_file in tqdm(image_files):
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file.replace(image_file[-4:], '_rgb_mask.png'))  # Assuming same filename for mask

        if not os.path.exists(mask_path):
            print(f"Mask for image '{image_file}' not found. Skipping.")
            continue

        try:
            # Open the original image and mask
            image = Image.open(image_path).convert("RGBA")
            mask = Image.open(mask_path).convert("RGBA")

            # Ensure both images are the same size
            if image.size != mask.size:
                # print(f"Size mismatch for '{image_file}'. Resizing mask to match the image.")
                mask = mask.resize(image.size, resample=Image.BILINEAR)

            # Adjust the mask's alpha channel based on the provided alpha value
            mask_with_alpha = mask.copy()
            alpha_mask = mask_with_alpha.split()[3].point(lambda p: p * alpha)
            mask_with_alpha.putalpha(alpha_mask)

            # Overlay the mask onto the image
            overlaid_image = Image.alpha_composite(image, mask_with_alpha)

            # Save the overlaid image
            output_path = os.path.join(output_folder, image_file)
            overlaid_image.convert("RGB").save(output_path)
            # print(f"Saved overlaid image to '{output_path}'.")

        except Exception as e:
            print(f"Failed to process '{image_file}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str)
    parser.add_argument('--masks', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()

    overlay_images_with_masks(args.images, args.masks, args.output, args.alpha)
