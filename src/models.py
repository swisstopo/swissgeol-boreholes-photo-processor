"""Module containing data models for the borehole image processing application."""

import re
from pathlib import Path

from PIL import Image


class ImageMetadata:
    """Class to represent metadata for an image."""

    def __init__(
        self, site_code: str, borehole_id: str, depth_start: float, depth_end: float, image_path: Path, folder: Path
    ):
        """Initializes an ImageMetadata instance.

        All fields are parsed from the image filename, which follows the convention
        ``[<site_code>][<borehole_id>][<depth_start>-<depth_end>]…``.

        Args:
            site_code: 1st bracket group in the filename, e.g. ``"GBC"``.
            borehole_id: 2nd bracket group in the filename, e.g. ``"CB50"``.
            depth_start: Start of the depth interval (metres), 3rd bracket group
                before the ``-``, e.g. ``15.0``.
            depth_end: End of the depth interval (metres), 3rd bracket group
                after the ``-``, e.g. ``16.0``.
            image_path: Full filesystem path to the image file,
                e.g. ``Path(".../[GBC]...TIF")``.
            folder: Parent directory used for batch routing,
                e.g. ``Path(".../CB50/")``.
        """
        self.site_code = site_code
        self.borehole_id = borehole_id
        self.depth_start = depth_start
        self.depth_end = depth_end
        self.image_path = image_path
        self.folder = folder

    _FILENAME_PATTERN = re.compile(
        r"^\[(?P<site_code>[^\]]+)\]"
        r"\[(?P<borehole_id>[^\]]+)\]"
        r"\[(?P<depth_start>\d+(?:\.\d+)?)\]-\[(?P<depth_end>\d+(?:\.\d+)?)\]"
    )

    @classmethod
    def from_path(cls, image_path: Path) -> "ImageMetadata":
        """Construct an ImageMetadata by parsing the structured filename.

        Args:
            image_path: Path to an image file whose name follows the convention
                ``[<site_code>][<borehole_id>][<depth_start>]-[<depth_end>]…``.

        Raises:
            ValueError: If the filename does not match the expected pattern.
        """
        match = cls._FILENAME_PATTERN.match(image_path.name)
        if not match:
            raise ValueError(f"Filename does not match expected pattern: {image_path.name}")
        return cls(
            site_code=match.group("site_code"),
            borehole_id=match.group("borehole_id"),
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
            site_code=metadata.site_code,
            borehole_id=metadata.borehole_id,
            depth_start=metadata.depth_start,
            depth_end=metadata.depth_end,
            image_path=metadata.image_path,
            folder=metadata.folder,
        )
        self.detections = detections
        self.bounding_boxes = bounding_boxes
