# GeoXMate Modeling Engine

A Python-based 3D geological modeling workflow designed for geoscientists.
This project enables structural modeling, facies modeling, simulation, seismic integration, and grid quality control using a modular and reproducible approach.

---

## 🚀 Features

* 🧱 **Structural Grid Generation** (layered 3D grids from horizons)
* 🗺️ **ZMAP Surface Processing** (loading & interpolation)
* 🪨 **Facies Upscaling** from well data
* 🎲 **Sequential Indicator Simulation (SIS)**
* 📡 **Seismic Attribute Mapping** (cube → grid)
* 📐 **Grid Quality Control** (cell angle analysis)
* 📊 **Interactive Visualization** using PyVista
* 📓 **Notebook-based workflow** (no GUI dependency)

---

## 📦 Project Structure

```
GeoXMate-Modeling/
│
├── src/
│   ├── grid/              # Grid construction
│   ├── data_io/           # ZMAP reader & interpolation
│   ├── modeling/          # Facies, simulation, seismic
│   ├── qc/                # Grid quality control
│   └── visualization/     # PyVista visualization
│
├── notebooks/
│   └── 01_structural_modeling.ipynb
│
├── data/                  # (Optional) sample data
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/khaled1240274/geoxmate-modeling.git
cd geoxmate-modeling
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

**Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Usage

Open the main notebook:

```
notebooks/01_structural_modeling.ipynb
```

This notebook demonstrates the full workflow:

1. Build grid
2. Load horizons (ZMAP)
3. Upscale facies
4. Run SIS simulation
5. Map seismic data
6. Perform QC
7. Visualize results

---

## 📊 Example Workflow

```python
from grid.geogrid import GeoGrid
from modeling.facies import upscale_facies
from modeling.simulation import run_sis_simulation

grid = GeoGrid()
grid.set_parameters({...})
grid.build_xy_grid()
grid.build_3d_grid(Z_maps)

facies = upscale_facies(grid, "data/wells.csv")
grid.add_property("FACIES", facies)

facies_sim = run_sis_simulation(grid)
grid.add_property("FACIES_SIS", facies_sim)
```

---

## 📁 Data Requirements

### Horizons (ZMAP)

* ARG
* UB
* LB
* KH

### Well Data (CSV)

Must include:

* `Easting`
* `Northing`
* `TVDSS`
* `FACIES`

### Seismic (NetCDF / xarray)

Expected variables:

* `cdp_x`
* `cdp_y`
* `samples`
* `data`

---

## ⚠️ Notes

* Large datasets (seismic cubes, full wells) are **not included** in this repository.
* Use your own data or small demo datasets.
* Python **3.11 recommended** for full compatibility.

---

## 🧠 Technical Highlights

* Vectorized spatial indexing for performance
* KDTree-based spatial simulation
* Memory-safe seismic interpolation (chunked)
* Structured grid handling via PyVista
* Modular design for extensibility

---

## 📈 Future Improvements

* Machine Learning integration (facies prediction)
* Export to LAS / VTK / industry formats
* Advanced geostatistics (variogram-based SIS)
* Interactive slice viewer (Petrel-style)
* GPU acceleration

---

## 👤 Author

**Khaled Saleh**
Cairo University
Email: [khaledsaleh@gstd.sci.cu.edu.eg](mailto:khaledsaleh@gstd.sci.cu.edu.eg)

---

## 📜 License

This project is for research and educational use.
(You can update with a proper license if needed)

---

## ⭐ Acknowledgment

If you find this project useful, consider starring the repository ⭐
