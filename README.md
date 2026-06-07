# NeRF vs 3D Gaussian Splatting

Empirical comparison of **NeRF** (`nerfacto`) and **3D Gaussian Splatting**
(`splatfacto`) for novel view synthesis, built on
[nerfstudio](https://docs.nerf.studio/). Trains both methods on the same scene
under identical settings and reports training time, rendering speed (FPS), and
image quality (PSNR / SSIM / LPIPS).

Final-exam research project — CMP6008.

---

## 1. Requirements

- NVIDIA GPU (tested: RTX 4060 Laptop, **8 GB VRAM**)
- Recent NVIDIA driver with CUDA support (WSL2 works)
- [conda / Anaconda](https://www.anaconda.com/download)

This guide reproduces a verified Linux / WSL2 install. nerfstudio's fast
extensions (`tiny-cuda-nn`) must be **compiled against CUDA 11.8**, so we build
an isolated environment with its own CUDA toolkit and a matching compiler.

---

## 2. Install nerfstudio

### 2.1 Create the environment

```bash
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
pip install --upgrade pip
```

### 2.2 PyTorch 2.1.2 + CUDA 11.8

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"     # torch 2.1.2 is built against NumPy 1.x
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 4060 Laptop GPU
```

### 2.3 CUDA 11.8 toolkit + matching compiler

The toolkit provides `nvcc`. CUDA 11.8 needs gcc ≤ 11 (system gcc is often
too new), so we install gcc 11 into the env as the host compiler.

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11 -y
```

### 2.4 Build tiny-cuda-nn (NeRF acceleration)

```bash
pip install ninja "setuptools<70" wheel   # setuptools<70 keeps pkg_resources

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
export TCNN_CUDA_ARCHITECTURES=89          # 89 = Ada / RTX 40-series; see note below
export LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

pip install --no-build-isolation \
  git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

> **GPU architecture:** `89` is for Ada (RTX 40-series). Use `86` for Ampere
> (RTX 30-series), `75` for Turing (RTX 20-series).

### 2.5 nerfstudio

```bash
pip install nerfstudio
```

### 2.6 WSL note (important)

On WSL2 the CUDA driver library lives in a non-standard path
(`/usr/lib/wsl/lib`). Make conda set it automatically on every activation:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' \
  > $CONDA_PREFIX/etc/conda/activate.d/wsl_cuda.sh
```

Re-activate the env afterwards (`conda activate nerfstudio`). On native Linux
this step is not needed.

### 2.7 Verify the install

```bash
python -c "import tinycudann; print('tinycudann OK')"
python -c "import gsplat; print('gsplat', gsplat.__version__)"
ns-train --help | grep -oE 'nerfacto|splatfacto|instant-ngp'
```

---

## 3. Get a dataset

We use **Mip-NeRF 360** — the benchmark used in both the Mip-NeRF and 3D
Gaussian Splatting papers, so results are comparable to published numbers. It
ships with a COLMAP reconstruction, so **no COLMAP install is required**.

```bash
mkdir -p data && cd data
wget "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip" -O 360.zip
unzip -q 360.zip -d mipnerf360
rm 360.zip
cd ..
```

This gives `data/mipnerf360/flowers` and `data/mipnerf360/treehill` (173 images
each, pre-downscaled `images_2/4/8`, COLMAP at `sparse/0/`). For the full
7-scene bundle (`bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`,
`stump`) download `360_v2.zip` from the same URL base instead.

> nerfstudio's built-in `ns-download-data nerfstudio` often fails with Google
> Drive quota errors — the direct download above avoids that.

---

## 4. Run the comparison

```bash
conda activate nerfstudio
python scripts/run_comparison.py
```

This trains `nerfacto` (NeRF) and `splatfacto` (3DGS) on `flowers` at 30k
iterations, evaluates both, and prints a results table. Outputs land in
`outputs/flowers/`:

| File | Contents |
|---|---|
| `comparison.md`  | results table (paste into the report) |
| `comparison.csv` | same data, machine-readable |
| `timings.csv`    | wall-clock training time per method |
| `<method>_metrics.json` | raw PSNR / SSIM / LPIPS / FPS from `ns-eval` |

### Options

```bash
python scripts/run_comparison.py --scene treehill
python scripts/run_comparison.py --methods nerfacto splatfacto instant-ngp tensorf
python scripts/run_comparison.py --iters 5000        # quick draft run
python scripts/run_comparison.py --downscale 4       # lower res -> less VRAM
python scripts/run_comparison.py --summarize-only    # rebuild table, no retrain
```

`--downscale 4` keeps memory within 8 GB VRAM. Reduce iterations for faster
draft runs; use the defaults for report-quality numbers.

---

## 5. Inspect a trained model

```bash
# Interactive web viewer (prints a localhost URL to open in a browser)
ns-viewer --load-config outputs/flowers/splatfacto/run/config.yml

# Render a camera-path video / images
ns-render --load-config outputs/flowers/nerfacto/run/config.yml
```

The viewer is useful for grabbing figures and for the live demo in the
presentation.

---

## 6. Train a single method manually

```bash
ns-train nerfacto \
  --data data/mipnerf360/flowers --output-dir outputs \
  colmap --colmap-path sparse/0 --downscale-factor 4

ns-train splatfacto \
  --data data/mipnerf360/flowers --output-dir outputs \
  colmap --colmap-path sparse/0 --downscale-factor 4
```

---

## Appendix A — Native Windows install

The instructions above target Linux / WSL2. If you have WSL2 available, that is
the **recommended** path on a Windows machine. To install on **native Windows**
instead, the differences are:

### A.1 Install a C++ compiler first

Install **Visual Studio 2019** with the *"Desktop development with C++"*
workload (the Build Tools edition is enough). This must be done **before** CUDA
so the toolkit can register with it. CUDA 11.8 targets the VS 2019 compiler.

### A.2 Install CUDA 11.8

Install the [CUDA 11.8 toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive)
from NVIDIA's native Windows installer (recommended over the conda toolkit on
Windows, since building `tiny-cuda-nn` needs `cl.exe` and `nvcc` together).

### A.3 Build from the VS native tools prompt

Open **"x64 Native Tools Command Prompt for VS 2019"** (Start menu) — this puts
both `cl.exe` and `nvcc` on `PATH`. Then:

```bat
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio
pip install --upgrade pip
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"

pip install ninja "setuptools<70" wheel
set TCNN_CUDA_ARCHITECTURES=89
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install nerfstudio
```

Notes vs. the Linux steps:

- **Do not** install the `gcc_linux-64` / `gxx_linux-64` conda packages — those
  are Linux-only; MSVC is the compiler on Windows.
- **Skip §2.6** (the WSL `libcuda` activation hook) — not needed on native
  Windows.
- Use `set VAR=value` (cmd) instead of `export VAR=...`. Set the GPU arch
  (`89`/`86`/`75`) the same way.
- If `tiny-cuda-nn` fails to find the compiler, you are likely in a plain
  terminal — reopen the **x64 Native Tools** prompt.

### A.4 Download the dataset (PowerShell)

```powershell
mkdir data; cd data
Invoke-WebRequest "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip" -OutFile 360.zip
Expand-Archive 360.zip -DestinationPath mipnerf360
Remove-Item 360.zip; cd ..
```

### A.5 Run

Identical to Linux — the comparison script is cross-platform:

```bat
conda activate nerfstudio
python scripts\run_comparison.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot find -lcuda` (build) or libcuda load error (runtime) | Ensure `LD_LIBRARY_PATH` includes `/usr/lib/wsl/lib` (see §2.6). |
| `No module named 'pkg_resources'` building tiny-cuda-nn | `pip install "setuptools<70"` and build with `--no-build-isolation`. |
| NumPy `_ARRAY_API not found` warning | `pip install "numpy<2"`. |
| CUDA out of memory | Increase `--downscale` (e.g. 8) or lower `--iters`. |
| `ns-download-data` fails with a Google Drive error | Use the direct dataset download in §3. |
| (Windows) `cl.exe` not found / compiler errors building tiny-cuda-nn | Open the **x64 Native Tools Command Prompt for VS 2019** and run the install from there (Appendix A.3). |
| (Windows) build fails with very long path errors | Enable long paths: `git config --global core.longpaths true` and enable Win32 long paths in Group Policy. |

---

## Project layout

```
cmp6008/
├── README.md
├── data/mipnerf360/{flowers,treehill}/   # datasets (not committed)
├── outputs/                              # training runs + result tables
└── scripts/
    └── run_comparison.py                 # train + eval + tabulate
```
