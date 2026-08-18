"""Module for preprocessing the cuttings images."""

import glob
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from fastembed import ImageEmbedding
from PIL import Image
from tqdm import tqdm

from src.config import SegmentationError
from src.mlflow_utils import log_artifact_with_mlflow, log_collect_cuttings_results_with_mlflow
from src.models import ImageMetadataCuttings

_CUTTINGS_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_grid(paths, cols=None, cell=(300, 300), pad=8, bg=(255, 255, 255)):
    """TODO.

    Args:
        paths (_type_): _description_
        cols (_type_, optional): _description_. Defaults to None.
        cell (tuple, optional): _description_. Defaults to (300, 300).
        pad (int, optional): _description_. Defaults to 8.
        bg (tuple, optional): _description_. Defaults to (255, 255, 255).

    Returns:
        _type_: _description_
    """
    imgs = [Image.open(p).convert("RGB") for p in paths]
    for im in imgs:
        im.thumbnail(cell)  # in place, keeps aspect ratio

    cols = cols or math.ceil(math.sqrt(len(imgs)))
    rows = math.ceil(len(imgs) / cols)

    w = cols * cell[0] + pad * (cols + 1)
    h = rows * cell[1] + pad * (rows + 1)
    sheet = Image.new("RGB", (w, h), bg)

    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        # centre each image inside its cell
        x = pad + c * (cell[0] + pad) + (cell[0] - im.width) // 2
        y = pad + r * (cell[1] + pad) + (cell[1] - im.height) // 2
        sheet.paste(im, (x, y))
    return sheet


def _select_by_similarity(
    samples: list[ImageMetadataCuttings], vector_samples: list[np.ndarray]
) -> tuple[list[ImageMetadataCuttings], dict[float, list]]:
    S = np.stack(vector_samples) @ np.stack(vector_samples).T
    id_ref = S.mean(axis=0).argmax()
    ids_sort = np.argsort(S[id_ref, :])[::-1]

    ranked_ids = {sample.depth: [] for sample in samples}
    ranked_filenames = {sample.depth: [] for sample in samples}
    for id_sort in ids_sort:
        img_metadata = samples[id_sort]
        ranked_ids[img_metadata.depth].append(id_sort.item())
        ranked_filenames[img_metadata.depth].append(img_metadata.image_path.name)

    return [samples[id_[0]] for id_ in ranked_ids.values()], ranked_filenames


def _select_by_first(samples: list[ImageMetadataCuttings]) -> list[ImageMetadataCuttings]:
    deduped_metadata: list[ImageMetadataCuttings] = []
    duplicate_counts: dict[float, int] = defaultdict(int)
    seen_depths: set[float] = set()

    for sample in sorted(samples, key=lambda m: (m.depth, m.image_path.name)):
        if sample.depth in seen_depths:
            duplicate_counts[sample.depth] += 1
            continue
        seen_depths.add(sample.depth)
        deduped_metadata.append(sample)

    return deduped_metadata


def collect_cuttings(input_dir: Path, with_mlflow: bool = False) -> list[ImageMetadataCuttings]:
    """Collect cuttings images from a directory, sorted by depth parsed from their filenames.

    Only the first image (by filename) at each depth is kept; the rest are dropped as
    duplicates and their count is logged to MLflow when with_mlflow is set. "00-Vials-"
    files (e.g. GVL-1's sample-vial photos) are excluded outright: their depth-less names
    would otherwise parse as depth 0 and pollute the output.

    Args:
        input_dir (Path): Path to the directory containing raw cuttings photos.
        with_mlflow (bool): Whether to log duplicate-depth stats to MLflow.

    Returns:
        list[ImageMetadataCuttings]: One entry per depth, sorted by depth.
    """
    model = ImageEmbedding("Qdrant/clip-ViT-B-32-vision", cache_dir="./models")

    # Collect all cutting images from the input directory and parse filename metadata
    files = sorted(map(Path, glob.glob(str(input_dir / "*"), include_hidden=False)))
    imgs_metadata: list[ImageMetadataCuttings] = []
    vectors: list[np.ndarray] = []

    for f in tqdm(files, desc="Collect cuttings ..."):
        if f.name.lower().startswith("00-vials-"):
            continue
        if f.suffix.lower() in _CUTTINGS_EXTENSIONS:
            try:
                metadata = ImageMetadataCuttings.from_path(f)
                metadata.borehole_id = input_dir.name
                _ = metadata.shape  # validate the file is readable before segmentation runs
                imgs_metadata.append(metadata)
                vectors.extend(list(model.embed(f)))
            except (ValueError, OSError, tifffile.TiffFileError, SegmentationError) as e:
                logging.warning("Skipping %s: %s", f.name, e)

    imgs_metadata_sim, ranked_filenames = _select_by_similarity(
        samples=imgs_metadata,
        vector_samples=vectors,
    )
    imgs_metadata_first = _select_by_first(samples=imgs_metadata)

    if with_mlflow:
        log_collect_cuttings_results_with_mlflow(ranked_filenames)
        log_artifact_with_mlflow(make_grid([d.image_path for d in imgs_metadata_sim]), "similarity")
        log_artifact_with_mlflow(make_grid([d.image_path for d in imgs_metadata_first]), "first")

    logging.info("Found %d cuttings images in %s", len(imgs_metadata_sim), input_dir.name)

    return sorted(imgs_metadata_sim, key=lambda x: x.depth)
