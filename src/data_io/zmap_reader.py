import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# --------------------------------------------------
# ZMAP READER
# --------------------------------------------------
def read_zmap(filepath):
    """
    Read ZMAP grid file and return a DataFrame with:
    Easting, Northing, Depth

    Assumptions:
    - Standard ZMAP format
    - Depth values (will be forced negative)
    """

    with open(filepath, "r") as f:
        lines = f.readlines()

    # -----------------------------
    # Find header
    # -----------------------------
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("@Grid HEADER"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Invalid ZMAP: '@Grid HEADER' not found")

    # -----------------------------
    # Parse header values
    # -----------------------------
    try:
        var_line = lines[header_idx + 1].strip().split(",")
        null_value = float(var_line[1])

        bounds_line = lines[header_idx + 2].strip().split(",")
        ny = int(bounds_line[0])
        nx = int(bounds_line[1])

        x_min = float(bounds_line[2])
        x_max = float(bounds_line[3])
        y_min = float(bounds_line[4])
        y_max = float(bounds_line[5])

    except Exception as e:
        raise ValueError(f"Error parsing ZMAP header: {e}")

    # -----------------------------
    # Find data start (2nd '@')
    # -----------------------------
    at_count = 0
    data_start = None

    for i, line in enumerate(lines):
        if line.strip().startswith("@"):
            at_count += 1
            if at_count == 2:
                data_start = i + 1
                break

    if data_start is None:
        raise ValueError("Invalid ZMAP: data section not found")

    # -----------------------------
    # Read data
    # -----------------------------
    data_values = []

    for line in lines[data_start:]:
        parts = line.strip().split()
        for val in parts:
            try:
                data_values.append(float(val))
            except:
                continue

    data_values = np.array(data_values)

    if len(data_values) != nx * ny:
        raise ValueError(
            f"ZMAP size mismatch: expected {nx*ny}, got {len(data_values)}"
        )

    # -----------------------------
    # Build grid
    # -----------------------------
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    XX, YY = np.meshgrid(x, y, indexing="ij")

    # ZMAP is usually row-major → reshape carefully
    Z = data_values.reshape((nx, ny), order="C")

    # Reverse Y to match geological convention
    YY = YY[:, ::-1]
    Z = Z[:, ::-1]

    # -----------------------------
    # Build DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "Easting": XX.ravel(),
        "Northing": YY.ravel(),
        "Depth": Z.ravel()
    })

    # Handle null values
    df.replace(null_value, np.nan, inplace=True)

    # Force depth negative (TVDSS convention)
    df["Depth"] = -np.abs(df["Depth"])

    return df


# --------------------------------------------------
# INTERPOLATION
# --------------------------------------------------
def interpolate_zmap_to_grid_xy(XX, YY, df_zmap, method="linear"):
    """
    Interpolate irregular ZMAP points onto model grid.

    Parameters:
    - XX, YY: model grid
    - df_zmap: DataFrame from read_zmap
    - method: 'linear', 'nearest', 'cubic'
    """

    if df_zmap.empty:
        raise ValueError("ZMAP DataFrame is empty")

    points = df_zmap[["Easting", "Northing"]].values
    values = df_zmap["Depth"].values

    target_points = np.column_stack((XX.ravel(), YY.ravel()))

    # -----------------------------
    # Main interpolation
    # -----------------------------
    Z = griddata(points, values, target_points, method=method)

    # -----------------------------
    # Fill NaNs using nearest
    # -----------------------------
    if np.any(np.isnan(Z)):
        Z_nearest = griddata(points, values, target_points, method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)

    return Z.reshape(XX.shape)