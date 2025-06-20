import argparse
import os.path
from glob import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm


def count_classes_int(mask):
    """Count unique classes in an inter mask with shape (H, W)."""
    class_presence = defaultdict(int)
    unique_classes = np.unique(mask)
    for cls in unique_classes:
        class_presence[cls] = 1
    return class_presence


def count_classes_binary(mask):
    """Count unique classes in a binary mask with shape (C, H, W)."""
    class_presence = defaultdict(int)
    num_classes = mask.shape[0]
    for cls in range(num_classes):
        class_presence[cls] = 1 if np.any(mask[cls] > 0) else 0
    return class_presence


def load_and_process_mask(file_path):
    """Load mask from .png or .npz file."""
    if file_path.endswith('.png'):
        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not load image from {file_path}")
        class_presence = count_classes_int(mask)
        return class_presence
    elif file_path.endswith('.npz'):
        data = np.load(file_path, mmap_mode='r')
        mask = data['arr']
        class_presence = count_classes_binary(mask)
        del data.f
        data.close()
        return class_presence
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def compute_class_statistics(mask_paths, max_workers):
    """Compute dataset statistics for masks in parallel with a progress bar."""
    overall_class_counts = defaultdict(int)

    with ThreadPoolExecutor(max_workers) as executor:
        futures = {executor.submit(load_and_process_mask, path): path for path in mask_paths}

        # Use tqdm for progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Masks"):
            path = futures[future]
            try:
                class_presence = future.result()
                for cls, presence in class_presence.items():
                    overall_class_counts[cls] += presence
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return dict(overall_class_counts)


def plot_class_statistics(stats1, stats2, filename):
    """Plot and save the class statistics for comparison using Seaborn with dual y-axes."""
    classes = sorted(set(stats1.keys()).union(set(stats2.keys())))
    counts1 = [stats1.get(cls, 0) for cls in classes]
    counts2 = [stats2.get(cls, 0) for cls in classes]

    x = np.arange(len(classes))  # the label locations
    width = 0.4  # the width of the bars

    # Create the figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for Dataset 1
    bars1 = ax1.bar(x - width / 2, counts1, width, label='Dataset 1', color='blue', alpha=0.6)
    ax1.set_ylabel('Counts for Dataset 1', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)

    # Create the second y-axis
    ax2 = ax1.twinx()

    # Bar plot for Dataset 2
    bars2 = ax2.bar(x + width / 2, counts2, width, label='Dataset 2', color='orange', alpha=0.6)
    ax2.set_ylabel('Counts for Dataset 2', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add title and grid
    ax1.set_title('Class Statistics Comparison')
    ax1.grid(axis='y')

    # Add legends for both datasets
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def main(path1: str, pattern1: str, path2: str, pattern2: str, plot_name: str, max_workers: int = 8):

    mask_files_png1 = natsorted(glob(os.path.join(path1, '**', '*.png'), recursive=True))
    mask_files_npz1 = natsorted(glob(os.path.join(path1, '**', '*.npz'), recursive=True))

    #assert not (len(mask_files_npz1) > 0 and len(mask_files_png1) > 0), f"Found .png AND .npz files in {path1}!"

    _mask_files1 = mask_files_png1 + mask_files_npz1
    mask_files1 = []
    for mask_file in tqdm(_mask_files1, 'Matching pattern1'):
        if pattern1 in mask_file:
            mask_files1.append(mask_file)

    print(f"Found {len(mask_files1)} masks in path1.")

    mask_files_png2 = natsorted(glob(os.path.join(path2, '**', '*.png'), recursive=True))
    mask_files_npz2 = natsorted(glob(os.path.join(path2, '**', '*.npz'), recursive=True))

    #assert not (len(mask_files_npz2) > 0 and len(mask_files_png2) > 0), f"Found .png AND .npz files in {path2}!"

    _mask_files2 = mask_files_png2 + mask_files_npz2
    mask_files2 = []
    for mask_file in tqdm(_mask_files2, 'Matching pattern2'):
        if pattern2 in mask_file:
            mask_files2.append(mask_file)

    print(f"Found {len(mask_files2)} masks in path2.")

    assert len(mask_files1) > 0
    assert len(mask_files2) > 0

    stats1 = compute_class_statistics(mask_files1, max_workers)
    stats2 = compute_class_statistics(mask_files2, max_workers)

    plot_class_statistics(stats1, stats2, plot_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, help='Path to 1st data folder (masks in .png OR .npz format).')
    parser.add_argument('--pattern1', type=str, help='Pattern to consider when looking for masks in path1.')
    parser.add_argument('--path2', type=str, help='Path to 2nd data folder (masks in .png OR .npz format).')
    parser.add_argument('--pattern2', type=str, help='Pattern to consider when looking for masks in path2.')
    parser.add_argument('--plot_name', type=str, help='Plot name for saving statistics visualisation.')
    parser.add_argument('--max_workers', type=int, help='Num. workers for multi-processing.', default=8)
    args = parser.parse_args()
    main(args.path1, args.pattern1, args.path2, args.pattern2, args.plot_name, args.max_workers)
