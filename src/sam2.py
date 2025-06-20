import torch
from sklearn.cluster import KMeans


def farthest_point_sampling(coords: torch.Tensor, n_a: int):
    """ Sample n_a points using farthest point sampling to spread points. """

    # Convert coords to float for distance computation
    coords = coords.float()

    # Initialize a tensor for the sampled points
    sampled_points = torch.empty((n_a, 2), dtype=torch.long, device=coords.device)

    # Start by picking a random point
    idx = torch.randint(0, len(coords), (1,))
    sampled_points[0] = coords[idx].long()

    # Distance tensor for all points
    dist = torch.full((len(coords),), float('inf'), device=coords.device)

    for i in range(1, n_a):
        # Update distances to the closest already selected point
        dist = torch.min(dist, torch.norm(coords - sampled_points[i - 1].float(), dim=1))
        # Select the farthest point
        farthest_idx = torch.argmax(dist)
        sampled_points[i] = coords[farthest_idx].long()

    return sampled_points


def kmeans_sampling(coords, n_a):
    """
    Sample n_a points by performing k-means clustering on the coordinates.
    The centroids of the clusters are returned as the sampled points.

    Args:
    coords (torch.Tensor): Coordinates of shape (N, 2), where N is the number of points.
    n_a (int): Number of points (clusters) to sample.

    Returns:
    sampled_points (torch.Tensor): Coordinates of sampled points (cluster centroids), shape (n_a, 2).
    """
    # Convert to numpy for k-means (KMeans in scikit-learn)
    coords_np = coords.cpu().numpy()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_a, random_state=0).fit(coords_np)

    # The cluster centers are the sampled points
    sampled_points_np = kmeans.cluster_centers_

    # Convert back to a PyTorch tensor (long for pixel coordinates)
    sampled_points = torch.from_numpy(sampled_points_np).long()

    return sampled_points


def sample_points_from_segmentation(segmentation_mask: torch.Tensor, n_a: int):
    """
    Given a segmentation mask of shape (K, H, W), sample n_a points from each class
    that are spread out as much as possible.

    Args:
    segmentation_mask (torch.Tensor): Segmentation mask of shape (K, H, W).
    n_a (int): Number of points to sample from each class.

    Returns:
    dict: A dictionary where keys are class indices and values are lists of sampled (H, W) coordinates.
    """
    K, H, W = segmentation_mask.shape
    sampled_points_dict = {}

    for k in range(K):
        # Get all coordinates for the current class
        class_mask = segmentation_mask[k] > 0  # Binary mask for class k
        coords = torch.nonzero(class_mask, as_tuple=False)  # Coordinates of non-zero points (pixels)

        if len(coords) < n_a:
            # raise ValueError(f"Not enough points to sample {n_a} points from class {k}")
            print(f"Not enough points to sample {n_a} points from class {k}")
            sampled_points_dict[k] = None

        else:
            # Sample n_a points using farthest point sampling
            sampled_points = kmeans_sampling(coords, n_a)

            sampled_points_dict[k] = sampled_points

    return sampled_points_dict