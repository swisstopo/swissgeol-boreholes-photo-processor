"""Module for preprocessing the cuttings images."""

from pathlib import Path

from src.models import ImageMetadataCuttings, ImageMetadataProcessedCuttings

# TODO: document in readme which file extensions we support
_CUTTINGS_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_cuttings(input_dir: Path) -> list[ImageMetadataProcessedCuttings]:
    """PLACEHOLDER FCT that collects cuttings images from a directory, sorted alphabetically by filename.

    Cuttings filenames carry a single point depth (e.g. "GES-F-1 195 m (Large).JPG") rather
    than the depth range ImageMetadataCores.from_path expects, so metadata is built directly
    here instead; depth is a placeholder index (unused by the cuttings grid layout).

    Args:
        input_dir (Path): Path to the directory containing raw cuttings photos.

    Returns:
        list[ImageMetadataProcessedCuttings]: One entry per cuttings image, in filename order.
    """
    image_paths = sorted(
        f for f in input_dir.iterdir() if not f.name.startswith("._") and f.suffix.lower() in _CUTTINGS_EXTENSIONS
    )
    return [
        ImageMetadataProcessedCuttings.from_metadata(
            ImageMetadataCuttings(borehole_id=input_dir.name, depth=float(i), image_path=path)
        )
        for i, path in enumerate(image_paths)
    ]


# TODO:
# 1. regex filter filenames
# 2. take only the first occurence
# 3. track number of images with the same depth
