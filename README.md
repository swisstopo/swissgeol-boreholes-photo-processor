# Boreholes Photo Processor

Boreholes Photo Processor is a pipeline that generates composite images from photos of borehole cores and cuttings. This project was initiated by the Swiss Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/), and is developed with support from [Visium](https://www.visium.com/).

## Installation

In a first step you need to clone the repository:

```bash
# using ssh
git clone git@github.com:swisstopo/swissgeol-boreholes-photo-processor.git

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

Then configure your credentials by running:

```bash
aws configure
```

this will prompt you for

```bash
AWS Access Key ID [None]: <your key id>
AWS Secret Access Key [None]: <your key>
Default region name [None]: eu-central-1
Default output format [None]: json
```

Now you can download the cores from the bucket:

```bash
aws s3 sync s3://stijnvermeeren-corephotos-cuttings/cores
```

And the cuttings from:

```bash
aws s3 sync s3://stijnvermeeren-corephotos-cuttings/cuttings
```