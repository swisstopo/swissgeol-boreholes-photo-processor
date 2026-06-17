"""Module containing data models for the borehole image processing application."""

import re
from pathlib import Path

from PIL import Image


class ImageMetadata:
    """Class to represent metadata for an image."""

    def __init__(self, borehole_id: str, depth_start: float, depth_end: float, image_path: Path, folder: Path):
        """Initializes an ImageMetadata instance.

        borehole_id is the filename prefix before the depth range.
        depth_start and depth_end are parsed from the filename, e.g.
        ``GBC-CB50_0015.00-0016.00_vd_p.TIF``.

        Args:
            borehole_id: Filename prefix before the depth range, e.g. ``"GBC-CB50"``.
            depth_start: Start of the depth interval (metres), e.g. ``15.0``.
            depth_end: End of the depth interval (metres), e.g. ``16.0``.
            image_path: Full filesystem path to the image file.
            folder: Parent directory of the image file, used to reconstruct the output path.
        """
        self.borehole_id = borehole_id
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.image_path = image_path
        self.folder = folder

    _DEPTH_PATTERN = re.compile(r"_(?P<depth_start>\d+\.\d+)-(?P<depth_end>\d+\.\d+)")

    @classmethod
    def from_path(cls, image_path: Path) -> "ImageMetadata":
        """Construct an ImageMetadata from an image path.

        borehole_id is extracted as the filename prefix before the depth range.
        depth_start and depth_end are extracted from the filename via regex.

        Args:
            image_path: Full path to an image file, e.g.
                ``Path(".../GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF")``.

        Raises:
            ValueError: If no depth range can be found in the filename.
        """
        match = cls._DEPTH_PATTERN.search(image_path.stem)
        if not match:
            raise ValueError(f"No depth range found in filename: {image_path.name}")
        return cls(
            borehole_id=image_path.stem[: match.start()],
            depth_start=float(match.group("depth_start")),
            depth_end=float(match.group("depth_end")),
            image_path=image_path,
            folder=image_path.parent,
        )


class ImageMetadataProcessed(ImageMetadata):
    """Class to represent metadata for a processed image."""

    def __init__(
        self,
        metadata: ImageMetadata,
        detections: list[Image.Image],
        bounding_boxes: list[tuple[float, float, float, float]],
    ):
        """Initializes an ImageMetadataProcessed instance.

        Args:
            metadata: The original image metadata.
            detections: Detected image regions produced by segmentation.
            bounding_boxes: Bounding box per detection as (x, y, width, height),
                used for drawing rectangles when plotting.
        """
        super().__init__(
            borehole_id=metadata.borehole_id,
            depth_start=metadata.depth_start,
            depth_end=metadata.depth_end,
            image_path=metadata.image_path,
            folder=metadata.folder,
        )
        self.detections = detections
        self.bounding_boxes = bounding_boxes
