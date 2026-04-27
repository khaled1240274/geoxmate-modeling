import numpy as np
import pyvista as pv


class GeoGrid:
    """
    Core 3D geological grid object.

    Responsibilities:
    - Define grid geometry
    - Build structured grid (PyVista)
    - Manage properties
    - Handle zones & layers indexing
    """

    def __init__(self):
        # Spatial limits
        self.X_min = None
        self.X_max = None
        self.Y_min = None
        self.Y_max = None
        self.Z_min = None
        self.Z_max = None

        # Resolution
        self.dx = None
        self.dy = None
        self.cell_thickness = None

        # Layering
        self.zones_config = None      # REQUIRED
        self.layers_config = None     # OPTIONAL

        # Geometry
        self.XX = None
        self.YY = None
        self.X3D = None
        self.Y3D = None
        self.Z3D = None

        # PyVista grid
        self.grid = None

        # Properties storage
        self.properties = {}

        # Cache last maps (for rebuild)
        self.last_Z_maps = None

    # --------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------
    def set_parameters(self, params: dict):
        required_keys = [
            "X_min", "X_max", "Y_min", "Y_max",
            "Z_min", "Z_max", "dx", "dy",
            "cell_thickness", "zones_config"
        ]

        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.X_min = params["X_min"]
        self.X_max = params["X_max"]
        self.Y_min = params["Y_min"]
        self.Y_max = params["Y_max"]
        self.Z_min = params["Z_min"]
        self.Z_max = params["Z_max"]

        self.dx = params["dx"]
        self.dy = params["dy"]
        self.cell_thickness = params["cell_thickness"]

        self.zones_config = params["zones_config"]
        self.layers_config = params.get("layers_config", None)

        self._validate_parameters()

    def _validate_parameters(self):
        if self.X_min >= self.X_max:
            raise ValueError("X_min must be < X_max")

        if self.Y_min >= self.Y_max:
            raise ValueError("Y_min must be < Y_max")

        if self.Z_min >= self.Z_max:
            raise ValueError("Z_min must be < Z_max")

        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("dx and dy must be positive")

    # --------------------------------------------------
    # GRID BUILDING
    # --------------------------------------------------
    def build_xy_grid(self):
        """
        Build 2D XY mesh grid
        """
        x = np.arange(self.X_min, self.X_max, self.dx)
        y = np.arange(self.Y_min, self.Y_max, self.dy)

        self.XX, self.YY = np.meshgrid(x, y, indexing='ij')

        return self.XX, self.YY

    def _resolve_layers(self):
        """
        Resolve layer configuration:
        - If user defined → use it
        - Else → 1 layer per zone
        """
        if self.layers_config is not None:
            return self.layers_config

        return {zone: 1 for zone in self.zones_config.keys()}

    def build_3d_grid(self, Z_maps: dict):
        """
        Build 3D structured grid from Z maps.

        Z_maps must contain:
        - ARG
        - UB
        - LB
        - KH
        """
        if self.XX is None or self.YY is None:
            raise RuntimeError("XY grid not built. Call build_xy_grid() first.")

        required_maps = ["ARG", "UB", "LB", "KH"]
        for key in required_maps:
            if key not in Z_maps:
                raise ValueError(f"Missing Z map: {key}")

        self.last_Z_maps = Z_maps

        XX, YY = self.XX, self.YY
        nx, ny = XX.shape

        layers_config = self._resolve_layers()

        if set(layers_config.keys()) != set(self.zones_config.keys()):
            raise ValueError("zones_config and layers_config mismatch")

        n_layers_total = sum(layers_config.values())
        nz = n_layers_total + 1

        X3D = np.zeros((nx, ny, nz))
        Y3D = np.zeros((nx, ny, nz))
        Z3D = np.zeros((nx, ny, nz))

        def build_layers(z_top, z_bot, n):
            return [z_top + i/n*(z_bot - z_top) for i in range(n+1)]

        for i in range(nx):
            for j in range(ny):

                z_stack = []

                z_stack += build_layers(self.Z_max, Z_maps["ARG"][i, j], layers_config["above_ARG"])[:-1]
                z_stack += build_layers(Z_maps["ARG"][i, j], Z_maps["UB"][i, j], layers_config["ARG_UB"])[:-1]
                z_stack += build_layers(Z_maps["UB"][i, j], Z_maps["LB"][i, j], layers_config["UB_LB"])[:-1]
                z_stack += build_layers(Z_maps["LB"][i, j], Z_maps["KH"][i, j], layers_config["LB_KH"])[:-1]
                z_stack += build_layers(Z_maps["KH"][i, j], self.Z_min, layers_config["below_KH"])

                X3D[i, j, :] = XX[i, j]
                Y3D[i, j, :] = YY[i, j]
                Z3D[i, j, :] = np.array(z_stack)

        self.X3D = X3D
        self.Y3D = Y3D
        self.Z3D = Z3D

        self.grid = pv.StructuredGrid(X3D, Y3D, Z3D)

        return self.grid

    # --------------------------------------------------
    # PROPERTIES
    # --------------------------------------------------
    def add_property(self, name, values):
        """
        Add cell-based property
        """
        expected_shape = (
            self.X3D.shape[0] - 1,
            self.X3D.shape[1] - 1,
            self.X3D.shape[2] - 1
        )

        if values.shape != expected_shape:
            raise ValueError(f"Property shape mismatch. Expected {expected_shape}, got {values.shape}")

        self.properties[name] = values
        self.grid.cell_data[name] = values.flatten(order='F')

    def get_property(self, name, flat=False, fill_value=None):
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found")

        data = self.properties[name]

        if fill_value is not None:
            data = np.where(np.isnan(data), fill_value, data)

        return data.flatten(order='F') if flat else data

    # --------------------------------------------------
    # INDEXING
    # --------------------------------------------------
    def get_layer_ids(self):
        nx, ny, nz = self.X3D.shape
        layer_ids = np.zeros((nx-1, ny-1, nz-1), dtype=int)

        for k in range(nz - 1):
            layer_ids[:, :, k] = k

        return layer_ids

    def get_zone_ids(self):
        layers_config = self._resolve_layers()

        nx, ny, nz = self.X3D.shape
        shape_cells = (nx-1, ny-1, nz-1)

        if sum(layers_config.values()) != shape_cells[2]:
            raise ValueError("Layer mismatch with grid")

        zone_ids = np.zeros(shape_cells, dtype=int)

        k_start = 0
        zone_id = 0

        for _, n_layers in layers_config.items():
            k_end = k_start + n_layers
            zone_ids[:, :, k_start:k_end] = zone_id
            k_start = k_end
            zone_id += 1

        return zone_ids

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------
    def rebuild(self):
        """
        Rebuild grid using last Z maps
        """
        if self.last_Z_maps is None:
            raise RuntimeError("No Z maps available for rebuild")

        return self.build_3d_grid(self.last_Z_maps)

    def get_parameters(self):
        return {
            "X_min": self.X_min,
            "X_max": self.X_max,
            "Y_min": self.Y_min,
            "Y_max": self.Y_max,
            "Z_min": self.Z_min,
            "Z_max": self.Z_max,
            "dx": self.dx,
            "dy": self.dy,
            "cell_thickness": self.cell_thickness,
            "layers_config": self.layers_config
        }