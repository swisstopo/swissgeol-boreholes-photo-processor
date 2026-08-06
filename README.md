# Boreholes Photo Processor

Boreholes Photo Processor is a pipeline that generates composite images from photos of borehole cores and cuttings. This project was initiated by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/), and is developed with support from [Visium](https://www.visium.com/).

## Installation

In a first step you need to clone the repository:

```bash
#using https
git clone https://github.com/swisstopo/swissgeol-boreholes-photo-processor.git
```

### OCR

The pipeline uses OCR to read the printed tick numbers on the depth ruler in each photo.

```bash
# Install Tesseract OCR engine (system-level)
# On Ubuntu/Debian:
sudo apt-get install -y tesseract-ocr=5.3.4*

# On macOS:
brew install tesseract

# On Windows: download installer from
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Python

We use [uv](https://docs.astral.sh/uv/) to manage package dependencies. Install uv first if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The below commands will install the package for you:
```bash
uv sync --all-extras
```

Then activate your environment:
```bash
source .venv/bin/activate
```

Adding packages can be done by editing the `pyproject.toml` of the project and adding the required package, then running `uv sync` to update the lock file.

## Running Tests

```bash
uv run pytest tests/
```

To also generate a coverage report:

```bash
uv run pytest --cov=src tests/
```

## Pipeline

### Download the borehole profiles, optional

You can download the data using the AWS CLI. First, you need to make sure that the CLI is installed:

```bash
brew install awscli
```

Configure your AWS credentials using `aws configure`, which will prompt you for:

```
AWS Access Key ID [None]: <your key id>
AWS Secret Access Key [None]: <your key>
Default region name [None]: eu-central-1
Default output format [None]: json
```

Now you can download the cores from the bucket:

```bash
aws s3 sync s3://stijnvermeeren-corephotos-cuttings/cores ./data/cores
```

And the cuttings from:

```bash
aws s3 sync s3://stijnvermeeren-corephotos-cuttings/cuttings ./data/cuttings
```

To sync only a single borehole (useful for local testing), specify the full prefix:

```bash
aws s3 sync s3://stijnvermeeren-corephotos-cuttings/cores/GBC/GBC-CB50 ./data/cores/GBC/GBC-CB50
```

### Expected Data Format (Cores)

The pipeline processes borehole core photos in **TIF format** (`.tif`/`.TIF`, case-insensitive). Files with other extensions are ignored.

**Filename convention**

Each image filename **must** contain a depth interval of the form `_XXXX.XX-YYYY.YY`, where `XXXX.XX` is the start depth and `YYYY.YY` is the end depth in metres. Files without this pattern are rejected at runtime.

The borehole identifier is derived from the filename prefix — everything before the depth range:

```
<borehole-id>_0015.00-0016.00_<optional-suffix>.tif
└───────────┘ └─────────────┘
 borehole_id    depth range
```

**Folder structure**

For a **single borehole**, the input directory should contain TIF files directly:

```
<any-folder>/
└── <borehole-folder>/          ← pass this as --input
    ├── <borehole-id>_0000.00-0001.00_*.tif
    ├── <borehole-id>_0001.00-0002.00_*.tif
    └── ...
```

For **batch processing** across multiple boreholes, the input directory should contain one subdirectory per borehole:

```
<any-folder>/                   ← pass this as --input
├── <borehole-folder-1>/
│   ├── <borehole-id-1>_0000.00-0001.00_*.tif
│   └── ...
└── <borehole-folder-2>/
    ├── <borehole-id-2>_0000.00-0001.00_*.tif
    └── ...
```

The pipeline detects the mode automatically: if the input directory contains subdirectories it runs in batch mode, otherwise it processes the directory as a single borehole. The input folder structure is mirrored in the output directory.

### Expected Data Format (Cuttings)

The pipeline processes cuttings photos in **JPG, JPEG, BMP, TIF, and TIFF format** (case-insensitive). Files with other extensions are ignored.

**Filename convention**

New data should follow this convention:

```
<borehole-id>_<depth>m_<sequence>.<ext>
```

e.g. `VINZEL-1_1234.50m_01.jpg` — `depth` is the point depth in metres (decimal allowed), and
`sequence` should be incremented for every extra photo taken at the same depth.

A number of legacy, borehole-specific filename conventions are also recognized for existing data
(e.g. an id followed by a depth range, an id and depth separated by a space, or a camera filename
with the depth appended as a trailing token) — see `ImageMetadataCuttings.from_path` in
`src/models.py` for the full list. Filenames with no recoverable depth are skipped with a warning,
not fatal to the whole batch. Files named `00-Vials-*` are always excluded outright, since they're
sample-vial photos rather than depth photos.

Only the first photo (by filename) found at a given depth is kept; any extra photos at the same
depth are dropped and the count is logged (and, with `--mlflow`, recorded as a metric). Naming every
photo at a shared depth with the convention above (unique, incrementing `sequence`) avoids this.

**Folder structure**

Same as for cores above: a single borehole folder of cuttings photos, or one subfolder per borehole
for batch processing.

### Output

Each output sheet places up to `num_cores_per_image` cores side by side, top-aligned on a black
background, with a ruler (major, intermediate, and minor ticks) drawn along both the left and
right edges, and the borehole ID printed in the top-left corner. Depth values are shown separately
as `depth_start`/`depth_end` labels (in metres) above and below each core strip.

The depth ruler is calibrated automatically: Tesseract OCR reads the printed tick numbers on each
photo's ruler to derive a pixels-per-unit scale (`px_per_unit`), so tick counts and labels reflect
actual detected depth rather than an arbitrary scale. Each core is then resized independently using
its own detected `px_per_unit` (falling back to the batch median if no ruler was detected for that image), and clamped if it would exceed `max_core_width`/`max_core_height`.

## Configuration

Segmentation, stitching and evaluation parameters are set via a YAML config file,
not CLI flags. A default [config.yaml](config.yaml) is provided at the repository root;
any omitted key falls back to its default (see `src/config.py`).

To speed up segmentation, images are downscaled before each detection step (tray, core-trim,
and ruler OCR each have their own `downscale_factor` under `segmentation.*` in `config.yaml`),
and resulting bounding boxes are scaled back up to the original resolution for stitching.

The pipeline groups images by their on-disk shape (height, width, channels) and, for each
group with at least `10` images, derives a shared bounding box by comparing images from the
(assumed static) camera position and locating the region that changes between shots (the
core/tray). Groups with fewer than `10` images fall back to per-image thresholding instead.
Per-image preprocessing for these shape groups (tray detection and ruler OCR) runs in parallel
across `segmentation.n_workers` worker processes, controlling which images are sampled from each group.

To use a different config file, pass `--config <path>` (see below).

## CLI Usage

### Run the pipeline for cores:

**Without MLflow tracking**

```bash
uv run boreholes-photo-processor --input <input-dir> --output <output-dir>
```

- `--input`: Path to the directory containing raw borehole photos (`.tif` only), or nested folders containing them
- `--output`: Path to the directory where processed images will be written
- `--config`: Path to the YAML config file for segmentation, stitching, and evaluation parameters (default: `config.yaml`)

**With MLflow tracking**

```bash
uv run boreholes-photo-processor --input <input-dir> --output <output-dir> --mlflow --debug
```

- `--mlflow`: Enable MLflow artifact logging. By default logs to `./mlruns`; set `MLFLOW_TRACKING_URI` for a remote server.
- `--debug`: Additionally log per-image and per-shape-group debug images (core/tray/ruler overlays) under a `debug` subfolder of each run's artifacts. Only has an effect when `--mlflow` is also set.


To view logged artifacts, start the MLflow UI:
```bash
uv run mlflow ui
```

Then open http://localhost:5000 in your browser.


### Run the pipeline for cuttings

**Without MLflow tracking**

```bash
uv run boreholes-photo-processor-cuttings --input <input-dir> --output <output-dir>
```

- same flags as for the cores

**With MLflow tracking**

```bash
uv run boreholes-photo-processor-cuttings --input <input-dir> --output <output-dir> --mlflow --debug
```

- same flags as for the cores
