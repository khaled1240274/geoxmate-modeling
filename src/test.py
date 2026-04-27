import sys
import os

sys.path.append(os.path.abspath("../src"))

from data_io.zmap_reader import read_zmap, interpolate_zmap_to_grid_xy

print("ZMAP module loaded ✅")