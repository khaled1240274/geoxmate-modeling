import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def run_sis_simulation(grid_obj, mode="layer", n_neighbors=5, random_seed=42):
    """
    Sequential Indicator Simulation (SIS)

    Parameters:
    - grid_obj: GeoGrid instance
    - mode: 'layer', 'zone', or 'global'
    - n_neighbors: number of nearest conditioning points
    - random_seed: for reproducibility

    Returns:
    - simulated facies grid (nx, ny, nz)
    """

    np.random.seed(random_seed)

    # -----------------------------
    # Get facies
    # -----------------------------
    facies = grid_obj.get_property("FACIES", flat=True)

    Xc = grid_obj.X3D[:-1, :-1, :-1]
    Yc = grid_obj.Y3D[:-1, :-1, :-1]
    Zc = grid_obj.Z3D[:-1, :-1, :-1]

    nx, ny, nz = Xc.shape

    Xi = Xc.flatten(order='F')
    Yi = Yc.flatten(order='F')
    Zi = Zc.flatten(order='F')

    simulated = facies.copy()

    # -----------------------------
    # Define subsets
    # -----------------------------
    if mode == "zone":
        ids = grid_obj.get_zone_ids().flatten(order='F')
    elif mode == "layer":
        ids = grid_obj.get_layer_ids().flatten(order='F')
    else:
        ids = np.zeros_like(simulated)

    unique_ids = np.unique(ids)

    # -----------------------------
    # Simulation loop
    # -----------------------------
    for uid in unique_ids:

        subset_idx = np.where(ids == uid)[0]

        known_mask = (~np.isnan(simulated)) & (ids == uid)

        X_known = Xi[known_mask]
        Y_known = Yi[known_mask]
        Z_known = Zi[known_mask]
        V_known = simulated[known_mask]

        if len(V_known) == 0:
            continue

        # Build KDTree
        tree = cKDTree(np.column_stack((X_known, Y_known, Z_known)))

        facies_unique = np.unique(V_known)

        # Random simulation path
        sim_indices = subset_idx.copy()
        np.random.shuffle(sim_indices)

        for idx in tqdm(sim_indices, desc=f"SIS ({mode} {uid})", unit="cell"):

            if not np.isnan(simulated[idx]):
                continue

            x, y, z = Xi[idx], Yi[idx], Zi[idx]

            k = min(n_neighbors, len(V_known))
            dist, ind = tree.query([x, y, z], k=k)

            if k == 1:
                dist = np.array([dist])
                ind = np.array([ind])

            Vn = V_known[ind]

            # -----------------------------
            # Inverse distance weighting
            # -----------------------------
            dist = np.maximum(dist, 1e-6)
            weights = 1.0 / dist
            weights /= weights.sum()

            probs = np.array([
                weights[Vn == f].sum() for f in facies_unique
            ])

            # Normalize
            if probs.sum() == 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs /= probs.sum()

            # Draw facies
            simulated[idx] = np.random.choice(facies_unique, p=probs)

            # 🔥 Update conditioning set (IMPORTANT for SIS)
            X_known = np.append(X_known, x)
            Y_known = np.append(Y_known, y)
            Z_known = np.append(Z_known, z)
            V_known = np.append(V_known, simulated[idx])

            tree = cKDTree(np.column_stack((X_known, Y_known, Z_known)))

    return simulated.reshape((nx, ny, nz), order='F')
