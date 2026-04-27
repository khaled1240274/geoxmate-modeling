import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


def map_seismic_to_grid(grid_obj, cube_path, chunk_size=500_000):
    """
    Map seismic cube values onto grid cells.

    Parameters:
    - grid_obj: GeoGrid instance
    - cube_path: path to seismic cube (NetCDF / xarray readable)
    - chunk_size: number of points per interpolation batch

    Returns:
    - amp_cells (nx, ny, nz)
    """

    if grid_obj.X3D is None:
        raise ValueError("Grid not built")

    # -----------------------------
    # Load seismic cube
    # -----------------------------
    ds = xr.open_dataset(cube_path)

    try:
        cdp_x = ds["cdp_x"].values
        cdp_y = ds["cdp_y"].values
        z_seis = -ds["samples"].values[::-1]
        seis = ds["data"].values[:, :, ::-1]
    except KeyError as e:
        raise ValueError(f"Missing expected variable in dataset: {e}")

    x_seis = cdp_x[:, 0]
    y_seis = cdp_y[0, :]

    # -----------------------------
    # Interpolator
    # -----------------------------
    interp = RegularGridInterpolator(
        (x_seis, y_seis, z_seis),
        seis,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    # -----------------------------
    # Grid points (nodes)
    # -----------------------------
    X3D, Y3D, Z3D = grid_obj.X3D, grid_obj.Y3D, grid_obj.Z3D

    points_model = np.column_stack([
        X3D.ravel(order="F"),
        Y3D.ravel(order="F"),
        Z3D.ravel(order="F")
    ])

    total_points = len(points_model)
    amp_model_flat = np.empty(total_points)

    n_chunks = (total_points + chunk_size - 1) // chunk_size

    # -----------------------------
    # Chunked interpolation
    # -----------------------------
    for i in tqdm(range(n_chunks), desc="Seismic Mapping", unit="chunk"):

        start = i * chunk_size
        end = min(start + chunk_size, total_points)

        chunk = points_model[start:end]

        amp_model_flat[start:end] = interp(chunk)

    # -----------------------------
    # Reshape back to grid (nodes)
    # -----------------------------
    amp_model = amp_model_flat.reshape(X3D.shape, order="F")

    # -----------------------------
    # Node → Cell averaging
    # -----------------------------
    amp_cells = (
        amp_model[:-1, :-1, :-1] +
        amp_model[1:, :-1, :-1] +
        amp_model[:-1, 1:, :-1] +
        amp_model[:-1, :-1, 1:] +
        amp_model[1:, 1:, :-1] +
        amp_model[1:, :-1, 1:] +
        amp_model[:-1, 1:, 1:] +
        amp_model[1:, 1:, 1:]
    ) / 8.0

    return amp_cells