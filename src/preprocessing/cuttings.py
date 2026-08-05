"""Module for preprocessing the cuttings images."""

from pathlib import Path

from src.models import ImageMetadata, ImageMetadataProcessed

_CUTTINGS_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_cuttings(input_dir: Path) -> list[ImageMetadataProcessed]:
    """PLACEHOLDER FCT that collects cuttings images from a directory, sorted alphabetically by filename.

    Cuttings filenames carry a single point depth (e.g. "GES-F-1 195 m (Large).JPG") rather
    than the depth range ImageMetadata.from_path expects, so metadata is built directly here
    instead; depth_start/depth_end are placeholders (unused by the cuttings grid layout).

    Args:
        input_dir (Path): Path to the directory containing raw cuttings photos.

    Returns:
        list[ImageMetadataProcessed]: One entry per cuttings image, in filename order.
    """
    image_paths = sorted(
        f for f in input_dir.iterdir() if not f.name.startswith("._") and f.suffix.lower() in _CUTTINGS_EXTENSIONS
    )
    return [
        ImageMetadataProcessed.from_metadata(
            ImageMetadata(borehole_id=input_dir.name, depth_start=float(i), depth_end=float(i + 1), image_path=path)
        )
        for i, path in enumerate(image_paths)
    ]
