import pyvista as pv
import numpy as np
from matplotlib.colors import ListedColormap


def facies_colormap():
    """
    Default categorical colormap for facies
    """
    return ListedColormap([
        "#66ff66",  # green
        "#cc9900",  # brown
        "#ffff00",  # yellow
        "#3399ff"   # blue
    ])

def show_grid(
    grid,
    scalar=None,
    cmap="viridis",
    show_edges=False,
    notebook=True,
    clim=None
):
    """
    Display grid in PyVista.

    Parameters:
    - grid: PyVista grid
    - scalar: property name (cell_data)
    - cmap: colormap
    - show_edges: show cell edges
    - notebook: use Jupyter backend
    - clim: [min, max]
    """

    if notebook:
        pv.set_jupyter_backend("client")

    plotter = pv.Plotter()

    if scalar is None:
        plotter.add_mesh(grid, show_edges=show_edges)

    else:
        if scalar not in grid.cell_data:
            raise ValueError(f"{scalar} not found in grid")

        values = grid.cell_data[scalar]

        plotter.add_mesh(
            grid,
            scalars=scalar,
            cmap=cmap,
            show_edges=show_edges,
            clim=clim
        )

    plotter.show_axes()
    plotter.add_bounding_box()
    plotter.view_isometric()

    plotter.show()
    