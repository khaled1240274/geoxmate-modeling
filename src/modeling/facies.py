import numpy as np
import pandas as pd


def upscale_facies(grid_obj, csv_path, facies_col="FACIES"):
    """
    Upscale well facies data to grid cells.

    Parameters:
    - grid_obj: GeoGrid instance
    - csv_path: path to well data CSV
    - facies_col: column name for facies

    Returns:
    - facies_grid (nx, ny, nz)
    """

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(csv_path)

    required_cols = ["Easting", "Northing", "TVDSS", facies_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # -----------------------------
    # Grid geometry
    # -----------------------------
    Xc = grid_obj.X3D[:-1, :-1, :-1]
    Yc = grid_obj.Y3D[:-1, :-1, :-1]
    Zc = grid_obj.Z3D[:-1, :-1, :-1]

    nx, ny, nz = Xc.shape

    # -----------------------------
    # Compute i, j (vectorized)
    # -----------------------------
    df = df.copy()

    df["i"] = ((df["Easting"] - grid_obj.X_min) / grid_obj.dx).astype(int)
    df["j"] = ((df["Northing"] - grid_obj.Y_min) / grid_obj.dy).astype(int)

    df["i"] = np.clip(df["i"], 0, nx - 1)
    df["j"] = np.clip(df["j"], 0, ny - 1)

    # -----------------------------
    # Compute k (optimized loop)
    # -----------------------------
    df["k"] = -1

    for (i, j), group in df.groupby(["i", "j"]):

        z_column = grid_obj.Z3D[i, j, :]

        # vectorized nearest depth index
        z_vals = group["TVDSS"].values[:, None]
        k_vals = np.argmin(np.abs(z_column - z_vals), axis=1)

        df.loc[group.index, "k"] = np.clip(k_vals, 0, nz - 1)

    # -----------------------------
    # Mode per cell
    # -----------------------------
    df_valid = df[df["k"] >= 0]

    def safe_mode(series):
        m = series.mode()
        return m.iloc[0] if len(m) > 0 else np.nan

    grouped = (
        df_valid
        .groupby(["i", "j", "k"])[facies_col]
        .apply(safe_mode)
        .reset_index()
    )

    # -----------------------------
    # Build grid
    # -----------------------------
    facies_grid = np.full((nx, ny, nz), np.nan)

    for _, row in grouped.iterrows():
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        facies_grid[i, j, k] = row[facies_col]

    return facies_grid
