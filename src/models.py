"""Module containing data models for the borehole image processing application."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PIL import Image


@dataclass
class ImageMetadata:
    """Metadata for an image file.

    borehole_id is the filename prefix before the depth range.
    depth_start and depth_end are parsed from the filename, e.g.
    ``GBC-CB50_0015.00-0016.00_vd_p.TIF``.
    """

    borehole_id: str
    depth_start: float
    depth_end: float
    image_path: Path

    _DEPTH_PATTERN: ClassVar[re.Pattern] = re.compile(r"_(?P<depth_start>\d+\.\d+)-(?P<depth_end>\d+\.\d+)")

    @classmethod
    def from_path(cls, image_path: Path) -> "ImageMetadata":
        """Construct an ImageMetadata from an image path.

        borehole_id is extracted as the filename prefix before the depth range.
        depth_start and depth_end are extracted from the filename via regex.

        Args:
            image_path (Path): Full path to an image file, e.g.
                ``Path(".../GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF")``.

        Raises:
            ValueError: If no depth range can be found in the filename.

        Returns:
            ImageMetadata: An instance containing the parsed metadata.
        """
        match = cls._DEPTH_PATTERN.search(image_path.stem)
        if not match:
            raise ValueError(f"No depth range found in filename: {image_path.name}")
        depth_start = float(match.group("depth_start"))
        depth_end = float(match.group("depth_end"))
        if depth_end <= depth_start:
            raise ValueError(
                f"depth_end ({depth_end}) must be greater than depth_start ({depth_start}) "
                f"in filename: {image_path.name}"
            )
        return cls(
            borehole_id=image_path.stem[: match.start()],
            depth_start=depth_start,
            depth_end=depth_end,
            image_path=image_path,
        )

    @property
    def folder(self) -> Path:
        """Parent directory of the image file."""
        return self.image_path.parent


@dataclass
class CoreSegmentResult:
    """Class to represent the result of processing a core segment image."""

    bounding_box: tuple[float, float, float, float]  # (left, upper, right, lower)


@dataclass
class ImageMetadataProcessed(ImageMetadata):
    """Metadata for a processed image with detected regions."""

    result: CoreSegmentResult

    @classmethod
    def from_metadata(
        cls,
        metadata: ImageMetadata,
        result: CoreSegmentResult,
    ) -> "ImageMetadataProcessed":
        """Construct an ImageMetadataProcessed from an existing ImageMetadata.

        Args:
            metadata (ImageMetadata): The original image metadata.
            result (CoreSegmentResult): The result of processing the image, e.g. bounding box and segmentation mask.

        Return:
            ImageMetadataProcessed: A new instance containing the original metadata and the processing result.
        """
        return cls(
            borehole_id=metadata.borehole_id,
            depth_start=metadata.depth_start,
            depth_end=metadata.depth_end,
            image_path=metadata.image_path,
            result=result,
        )

    def as_image(self) -> Image.Image:
        """Cut a core segment from the source image, rotating to portrait if needed.

        Cores are stored vertically in the output, so landscape crops (width > height)
        are rotated 90° clockwise so the left edge (shallow end) becomes the top.

        Returns:
            Image.Image: The cropped core segment image in portrait orientation.
        """
        with Image.open(self.image_path) as src:
            left, upper, right, lower = (round(v) for v in self.result.bounding_box)
            crop = src.crop((left, upper, right, lower))
            if crop.width > crop.height:
                crop = crop.transpose(Image.Transpose.ROTATE_270)  # clockwise: left (shallow) → top
        return crop
