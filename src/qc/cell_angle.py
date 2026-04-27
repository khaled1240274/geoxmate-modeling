import numpy as np
from tqdm import tqdm


def compute_cell_angle(grid, show_progress=True):
    """
    Compute deviation from orthogonality for each grid cell.

    Returns:
    - cell_angles (flattened array)
    """

    if grid is None:
        raise ValueError("Grid is not defined")

    points = grid.points
    nx, ny, nz = grid.dimensions

    # Convert to cell counts
    nx -= 1
    ny -= 1
    nz -= 1

    # -----------------------------
    # Helper functions
    # -----------------------------
    def get_point(i, j, k):
        return points[i + j*(nx+1) + k*(nx+1)*(ny+1)]

    def angle(u, v):
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)

        if nu == 0 or nv == 0:
            return 90.0  # safe fallback

        cos_theta = np.dot(u, v) / (nu * nv)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.degrees(np.arccos(cos_theta))

    # -----------------------------
    # Loop over cells
    # -----------------------------
    cell_angles = []

    iterator = range(nz)
    if show_progress:
        iterator = tqdm(iterator, desc="Cell Angle QC", unit="layer")

    for k in iterator:
        for j in range(ny):
            for i in range(nx):

                p0 = get_point(i, j, k)

                px = get_point(i+1, j, k)
                py = get_point(i, j+1, k)
                pz = get_point(i, j, k+1)

                vx = px - p0
                vy = py - p0
                vz = pz - p0

                a_xy = angle(vx, vy)
                a_xz = angle(vx, vz)
                a_yz = angle(vy, vz)

                # deviation from 90°
                dev = max(
                    abs(a_xy - 90),
                    abs(a_xz - 90),
                    abs(a_yz - 90)
                )

                cell_angles.append(dev)

    cell_angles = np.array(cell_angles)

    # -----------------------------
    # Attach to grid
    # -----------------------------
    grid.cell_data["CellAngle"] = cell_angles

    return cell_angles
