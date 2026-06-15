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

### Bucket Structure and Data Volumes

The S3 bucket `stijnvermeeren-corephotos-cuttings` is organized into two top-level prefixes:

| Category    | Files  | Size      | Primary Format |
|-------------|-------:|----------:|----------------|
| `cores/`    | 11,630 | 503.6 GiB | TIF            |
| `cuttings/` |  6,246 |  24.8 GiB | JPG            |

#### Core Photos (`cores/`)

Files are organized as `cores/<group>/<borehole>/`, covering 53 boreholes across 7 groups:

| Group | Boreholes | Files | Size (GiB) |
|---|---:|---:|---:|
| GBC | 12 | 3,527 | 133.2 |
| GBT | 12 | 1,938 | 108.0 |
| Georessourcen | 9 | 1,145 | 52.8 |
| Handstuecke | 1 | 121 | 5.4 |
| LBT | 12 | 2,839 | 114.6 |
| LBT_Prognose | 6 | 697 | 28.1 |
| VP_GBT | 1 | 1,362 | 61.5 |

#### Cuttings (`cuttings/`)

| Borehole | Files | JPG | TIF | HEIC | Other | Size (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Forsthaus GES-F1 | 40 | 33 | 0 | 0 | 7 | 0.20 |
| Forsthaus GES-F2 | 122 | 121 | 0 | 0 | 1 | 0.82 |
| Forsthaus GES-F3 | 193 | 192 | 0 | 0 | 1 | 1.25 |
| Forsthaus GES-F3A | 103 | 102 | 0 | 0 | 1 | 0.64 |
| GEo-01 | 629 | 628 | 0 | 0 | 1 | 1.42 |
| GEo-02 | 1,649 | 1,648 | 0 | 0 | 1 | 5.01 |
| GVL-1 | 1,250 | 1,206 | 0 | 43 | 1 | 5.84 |
| Lavey-1 ¹ | 2 | 0 | 0 | 0 | 2 | 0.08 |
| Montagny-2 | 254 | 253 | 0 | 0 | 1 | 1.21 |
| Montagny-2ST | 181 | 180 | 0 | 0 | 1 | 0.64 |
| Vinzel-1 | 996 | 786 | 209 | 0 | 1 | 5.52 |
| Vinzel-1-Malm | 552 | 551 | 0 | 0 | 1 | 1.32 |
| Vinzel-1S | 274 | 272 | 1 | 0 | 1 | 0.83 |

¹ Lavey-1 does not have original image files. Photos must be extracted from a large PDF file.

## CLI Usage

Run the borehole photo processing pipeline:

**Without MLflow tracking**

```bash
uv run boreholes-photo-processor --input <input-dir> --output <output-dir>
```

- `--input`: Path to the directory containing raw borehole photos (`.tif` only), or nested folders containing them
- `--output`: Path to the directory where processed images will be written

**With MLflow tracking**

```bash
uv run boreholes-photo-processor --input <input-dir> --output <output-dir> --mlflow
```

- `--mlflow`: Enable MLflow artifact logging. By default logs to `./mlruns`; set `MLFLOW_TRACKING_URI` for a remote server. Also auto-enabled when `MLFLOW_TRACKING_URI` is present in the environment.

To view logged artifacts, start the MLflow UI:
```bash
uv run mlflow ui
```

Then open http://localhost:5000 in your browser.

## Remote Execution on Azure ML

The pipeline can be submitted as a job to Azure ML, which runs it on a remote compute cluster with MLflow tracking configured automatically.

### Prerequisites

```bash
brew install azure-cli
az extension add -n ml
az login
```

### Submitting a job

The job is defined in `azure/job.yml`. Update the `inputs.borehole_data.path` field to point at your registered data asset, then submit:

```bash
az ml job create \
  --file azure/job.yml \
  --resource-group rg-swisstopo-compute \
  --workspace-name aml-swisstopo-sn
```

MLflow tracking is enabled automatically — no `--mlflow` flag needed.

### Monitoring results

Open [ml.azure.com](https://ml.azure.com) → workspace `aml-swisstopo-sn` → **Jobs** → your run:

- **Outputs + logs** → `user_logs/std_log.txt` for the full run log
- **Outputs** tab → `results/` folder for the output TIF and PNG files
- **Images** → MLflow artifacts (detection plots, stitched image)
