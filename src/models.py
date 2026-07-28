"""Module containing data models for the borehole image processing application."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from PIL import Image

from src.utils import get_image_shape, load_image


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
            ValueError: If no depth range can be found in the filename, or if the
                parsed depth_end is not strictly greater than depth_start.

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

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the image as (height, width, channels).

        Returns:
            tuple[int, int, int]: The shape of the image.
        """
        return get_image_shape(str(self.image_path))

    def load_image(self, factor: float = 1.0) -> np.ndarray:
        """Load a TIF image and normalize it to an RGB float array in [0, 1].

        Args:
            factor (float): Downscale factor applied after loading; 1.0 leaves the image unscaled.

        Returns:
            np.ndarray: RGB image array with float values in [0, 1].

        Raises:
            ValueError: If the image is not a 3-channel RGB array, or has an unsupported dtype.
        """
        return load_image(str(self.image_path), factor)


@dataclass
class SegmentationRecord:
    """Per-image record of which segmentation approach was used, for the mlflow summary log."""

    tray_approach: Literal["group", "single"]
    tray_group: str | None
    ruler_approach: Literal["group", "single"]
    ruler_group: str | None


@dataclass
class ImageSegmentResult:
    """Class to represent the result of detecting a region in an image."""

    bbox: tuple[float, float, float, float]  # (left, top, right, bottom)


@dataclass
class CoreSegmentResult(ImageSegmentResult):
    """Result of detecting core bbox, with core bbox segments for MLflow logging."""

    bbox_segments: list[tuple[float, float, float, float]] | None = None


@dataclass
class TraySegmentResult(ImageSegmentResult):
    """Result of detecting the shared tray/core bbox, with optional debug images for MLflow logging."""

    img_background: np.ndarray | None = None  # mean image across the batch
    img_foreground: np.ndarray | None = None  # per-pixel std map used to estimate the bbox
    img_downscale_factor: float | None = None  # downscale factor the debug images above are stored at


@dataclass
class RulerSegmentResult(ImageSegmentResult):
    """Result of detecting a depth ruler in an image via OCR on its printed number ticks."""

    px_per_unit: float  # pixel distance between two consecutive ruler unit ticks, at full image resolution
    bbox_units: list[tuple[float, float, float, float]]  # one bbox per detected ruler number


@dataclass
class ImageMetadataProcessed(ImageMetadata):
    """Metadata for a processed image with detected regions."""

    core: CoreSegmentResult | None = None
    tray: ImageSegmentResult | None = None
    ruler: RulerSegmentResult | None = None
    records: SegmentationRecord | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: ImageMetadata,
        core: CoreSegmentResult | None = None,
        tray: ImageSegmentResult | None = None,
        ruler: RulerSegmentResult | None = None,
        records: SegmentationRecord | None = None,
    ) -> "ImageMetadataProcessed":
        """Construct an ImageMetadataProcessed from an existing ImageMetadata.

        Args:
            metadata (ImageMetadata): The original image metadata.
            core (CoreSegmentResult | None): Detected core bounding box, if any.
            tray (ImageSegmentResult | None): Detected tray bounding box, if any.
            ruler (RulerSegmentResult | None): Detected ruler bounding box, if any.
            records (SegmentationRecord | None): Record of which segmentation approach (group vs.
                single image) was used for the tray and ruler detections, for the mlflow summary log.

        Returns:
            ImageMetadataProcessed: A new instance containing the original metadata and the processing result.
        """
        return cls(
            borehole_id=metadata.borehole_id,
            depth_start=metadata.depth_start,
            depth_end=metadata.depth_end,
            image_path=metadata.image_path,
            core=core,
            tray=tray,
            ruler=ruler,
            records=records,
        )

    def load_core(self) -> Image.Image:
        """Cut a core segment from the source image, rotating to portrait if needed.

        Cores are stored vertically in the output, so landscape crops (width > height)
        are rotated 90° clockwise so the left edge (shallow end) becomes the top.

        Returns:
            Image.Image: The cropped core segment image in portrait orientation.

        Raises:
            ValueError: If no core region was detected for this image.
        """
        if self.core is None:
            raise ValueError(f"No core region detected for image: {self.image_path}")

        with Image.open(self.image_path) as src:
            left, upper, right, lower = (round(v) for v in self.core.bbox)
            crop = src.crop((left, upper, right, lower))
            if crop.width > crop.height:
                crop = crop.transpose(Image.Transpose.ROTATE_270)  # clockwise: left (shallow) → top
        return crop
