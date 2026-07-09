# Boreholes Photo Processor

Boreholes Photo Processor is a pipeline that generates composite images from photos of borehole cores and cuttings. This project was initiated by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/), and is developed with support from [Visium](https://www.visium.com/).

## Installation

In a first step you need to clone the repository:

```bash
#using https
git clone https://github.com/swisstopo/swissgeol-boreholes-photo-processor.git
```

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

### Expected Data Format

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

## Configuration

Segmentation and stitching parameters (padding, canvas size, cores per sheet, thresholds, etc.) are set via a YAML config file, not CLI flags. A default [config.yaml](config.yaml) is provided at the repository root; any omitted key falls back to its default (see `src/config.py`).

To speed up segmentation, images are downscaled by `downscale_factor` before detecting the core, and the resulting bounding box is scaled back up to the original resolution for stitching.

The pipeline first tries to derive a single bounding box shared across the whole batch, by comparing all images from the (assumed static) camera position and locating the region that changes between shots (the core). This requires at least `10` successfully loaded images in the batch. Smaller batches, or images with inconsistent size, fall back to per-image thresholding instead.

To use a different config file, pass `--config <path>` (see below).

## CLI Usage

Run the borehole photo processing pipeline:

**Without MLflow tracking**

```bash
uv run boreholes-photo-processor --input <input-dir> --output <output-dir>
```

- `--input`: Path to the directory containing raw borehole photos (`.tif` only), or nested folders containing them
- `--output`: Path to the directory where processed images will be written
- `--config`: Path to the YAML config file for segmentation and stitching parameters (default: `config.yaml`)

**With MLflow tracking**

```bash
uv run boreholes-photo-processor --input <input-dir> --output <output-dir> --mlflow
```

- `--mlflow`: Enable MLflow artifact logging. By default logs to `./mlruns`; set `MLFLOW_TRACKING_URI` for a remote server. Segmentation debug images (per-image bounding-box overlays and the batch's shared foreground estimate are logged under a `debug` subfolder of each run's artifacts.


To view logged artifacts, start the MLflow UI:
```bash
uv run mlflow ui
```

Then open http://localhost:5000 in your browser.
