import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import sklearn.datasets

from tqdm import tqdm

# ===== Algorithm implementation (GPU version if CuPy is available, otherwise CPU version)

try:
    import cupy
    import cupyx.scipy.ndimage
    
    def multiresolution_transformation(points: np.ndarray, maximum_resolution: int, cycle_count: int, kernel_radius: int) -> np.ndarray:
        points = cupy.asarray(points, dtype=cupy.float32)

        for _ in tqdm(range(cycle_count)):

            resolution = maximum_resolution
            while resolution >= 2:

                # Compute density
                density, _, _ = cupy.histogram2d(points[:, 0], points[:, 1], bins=resolution, range=[[0, 1], [0, 1]])
                density = cupyx.scipy.ndimage.gaussian_filter(density, sigma=(2 * kernel_radius + 1) / 6.0, mode="reflect")
                density = cupy.pad(density, pad_width=1, mode="edge")

                # Compute deformation
                D00 = density[:-1, :-1]
                D01 = density[:-1, 1:]
                D10 = density[1:, :-1]
                D11 = density[1:, 1:]

                denominator = 4.0 * (D00 + D01 + D10 + D11 + 1e-8)
                deformation = cupy.stack([
                    D00 + D01 - D10 - D11,
                    D00 - D01 + D10 - D11
                ], axis=-1) / denominator[:, :, None] / resolution

                # Perform bilinear interpolation
                xy = points * resolution
                ixy = cupy.minimum(cupy.floor(xy).astype(cupy.int32), resolution - 1)

                uv = xy - ixy
                u = uv[:, 0]
                v = uv[:, 1]

                d00 = deformation[ixy[:, 0] + 0, ixy[:, 1] + 0]
                d01 = deformation[ixy[:, 0] + 0, ixy[:, 1] + 1]
                d10 = deformation[ixy[:, 0] + 1, ixy[:, 1] + 0]
                d11 = deformation[ixy[:, 0] + 1, ixy[:, 1] + 1]

                w00 = (1 - u) * (1 - v)
                w01 = (1 - u) * v
                w10 = u * (1 - v)
                w11 = u * v
            
                points += w00[:, None] * d00 + w01[:, None] * d01 + w10[:, None] * d10 + w11[:, None] * d11
                points = cupy.clip(points, 0.0, 1.0)

                resolution //= 2

        return cupy.asnumpy(points)

except ImportError:
    def multiresolution_transformation(points: np.ndarray, maximum_resolution: int, cycle_count: int, kernel_radius: int) -> np.ndarray:
        for _ in tqdm(range(cycle_count)):

            resolution = maximum_resolution
            while resolution >= 2:

                # Compute density
                density, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=resolution, range=[[0.0, 1.0], [0.0, 1.0]])
                density = scipy.ndimage.gaussian_filter(density, sigma=(2.0 * kernel_radius + 1.0) / 6.0, mode="reflect")
                density = np.pad(density, pad_width=1, mode="edge")

                # Compute deformation
                D00 = density[:-1, :-1]
                D01 = density[:-1, 1:]
                D10 = density[1:, :-1]
                D11 = density[1:, 1:]

                denominator = 4.0 * (D00 + D01 + D10 + D11 + 1e-8)
                deformation = np.stack([
                    D00 + D01 - D10 - D11,
                    D00 - D01 + D10 - D11
                ], axis=-1) / denominator[:, :, None] / resolution

                # Perform bilinear interpolation
                xy = points * resolution
                ixy = np.minimum(np.floor(xy).astype(np.int32), resolution - 1)

                uv = xy - ixy
                u = uv[:, 0]
                v = uv[:, 1]

                d00 = deformation[ixy[:, 0] + 0, ixy[:, 1] + 0]
                d01 = deformation[ixy[:, 0] + 0, ixy[:, 1] + 1]
                d10 = deformation[ixy[:, 0] + 1, ixy[:, 1] + 0]
                d11 = deformation[ixy[:, 0] + 1, ixy[:, 1] + 1]

                w00 = (1 - u) * (1 - v)
                w01 = (1 - u) * v
                w10 = u * (1 - v)
                w11 = u * v
            
                points += w00[:, None] * d00 + w01[:, None] * d01 + w10[:, None] * d10 + w11[:, None] * d11
                points = np.clip(points, 0.0, 1.0)

                resolution //= 2

        return points

# ===== Utility functions

def normalize_to_unit_square(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    center = (minimum + maximum) / 2.0
    extent = (maximum - minimum).max()

    points = (points - center) / extent + 0.5
    return np.clip(points, 0.0, 1.0)



# ===== Prepare datasets
datasets = []

points, labels = sklearn.datasets.make_circles(n_samples=2000, noise=0.05, factor=0.5, random_state=42)
points = normalize_to_unit_square(points)
datasets.append(("Circles", points, labels, 10.0))

points, labels = sklearn.datasets.make_moons(n_samples=10000, noise=0.05, random_state=42)
points = normalize_to_unit_square(points)
datasets.append(("Moons", points, labels, 1.0))

np.random.seed(42)
points_a = np.random.normal(loc=0.0, scale=0.05, size=(2500, 2)) + np.array([0.3, 0.3])
points_b = np.random.normal(loc=0.0, scale=0.05, size=(5000, 2)) + np.array([0.4, 0.75])
points_c = np.random.normal(loc=0.0, scale=0.05, size=(2500, 2)) + np.array([0.75, 0.55])
points = normalize_to_unit_square(np.vstack([points_a, points_b, points_c]))
labels = np.array([0] * len(points_a) + [1] * len(points_b) + [2] * len(points_c))
datasets.append(("Clusters", points, labels, 1.0))

colors = np.array(["#6A3D9A", "#FF7F00",  "#33A02B"], dtype=object)

# ===== Prepare algorithms

algorithms = [
    ("Original", lambda points: points),
    ("Multiresolution", lambda points: multiresolution_transformation(points, maximum_resolution=1024, cycle_count=50, kernel_radius=2))
]

# ===== Generate figures

def generate_teaser():
    points, labels = sklearn.datasets.make_moons(n_samples=300, noise=0.05, random_state=42)
    points = normalize_to_unit_square(points)

    points_transformed = multiresolution_transformation(points, maximum_resolution=1024, cycle_count=50, kernel_radius=2)

    t = points[:, 0].flatten()
    t = 3 * t**2 - 2 * t**3

    points_interpolated = t[:, None] * points_transformed + (1 - t[:, None]) * points

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(1, 1))
    axis.scatter(points_interpolated[:, 0], points_interpolated[:, 1], c=colors[labels], s=1.0, clip_on=False)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    figure.savefig("teaser.png", dpi=250)

def generate_example():
    nrows = len(algorithms)
    ncols = len(datasets)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))

    for row, (algorithm_name, algorithm) in enumerate(algorithms):
        for col, (dataset_name, points, labels, point_size) in enumerate(datasets):
            axis = axes[row, col]

            transformed_points = algorithm(points)
            axis.scatter(transformed_points[:, 0], transformed_points[:, 1], c=colors[labels], s=point_size)

            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks([])
            axis.set_yticks([])

            if row == 0:
                axis.set_title(dataset_name, fontsize=20, fontweight="bold", pad=10)
            if col == 0:
                axis.set_ylabel(algorithm_name, fontsize=20, fontweight="bold", labelpad=10)

    figure.tight_layout()
    figure.savefig("example.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    # generate_teaser()
    generate_example()