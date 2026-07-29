"""Module containing data models for the borehole image processing application."""

import re
from dataclasses import asdict, dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import ClassVar

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

    def to_dict(self) -> dict:
        """Return this metadata as a plain dict, e.g. for JSON serialization."""
        return asdict(self)


class ApproachType(IntEnum):
    """Approach used to detect a region: per-image (SINGLE) or shared across a shape group (GROUP)."""

    SINGLE = 0
    GROUP = auto()


@dataclass
class ImageSegmentResult:
    """Class to represent the result of detecting a region in an image."""

    bbox: tuple[float, float, float, float]  # (left, top, right, bottom)
    time: float | None = None  # Processing time
    approach: ApproachType = ApproachType.SINGLE  # Type of approach used

    @staticmethod
    def approach_to_json(results: list["ImageSegmentResult | None"]) -> dict[str, float]:
        """Summarize per-approach detection counts and average timing across a batch.

        Args:
            results (list[ImageSegmentResult | None]): Per-image detection results for one region
                (e.g. every tray detection across a batch). None entries mark images where
                detection failed.

        Returns:
            dict[str, float]: Counts of images that failed (n_as_fail), used the per-image
                single approach (n_as_single), or used a shared group approach (n_as_group);
                the number of distinct group detections reused (n_group); and the average
                processing time for single- and group-approach detections (time_single_avg,
                time_group_avg).
        """
        ts_group = set([result.time or 0 for result in results if result and result.approach == ApproachType.GROUP])
        ts_single = [result.time or 0 for result in results if result and result.approach == ApproachType.SINGLE]
        n_fail = sum([result is None for result in results])

        return {
            "n_group": len(ts_group),
            "n_as_fail": n_fail,
            "n_as_single": len(ts_single),
            "n_as_group": len(results) - len(ts_single) - n_fail,
            "time_single_avg": sum(ts_single) / (len(ts_single) + 1e-16),
            "time_group_avg": sum(ts_group) / (len(ts_group) + 1e-16),
        }

    def to_dict(self) -> dict:
        """Return this result as a plain dict, e.g. for JSON serialization."""
        return {"approach": self.approach.name, "bbox": self.bbox, "time": self.time}


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

    def to_dict(self) -> dict:
        """Return this result as a plain dict, including the debug images' downscale factor."""
        return {**super().to_dict(), "img_downscale_factor": self.img_downscale_factor}


@dataclass
class RulerSegmentResult(ImageSegmentResult):
    """Result of detecting a depth ruler in an image via OCR on its printed number ticks."""

    px_per_unit: float | None = None  # pixel distance between two consecutive ruler unit ticks
    bbox_units: list[tuple[float, float, float, float]] | None = None  # one bbox per detected ruler number

    def to_dict(self) -> dict:
        """Return this result as a plain dict, including the pixel-per-unit scale."""
        return {**super().to_dict(), "px_per_unit": self.px_per_unit}


@dataclass
class ImageMetadataProcessed(ImageMetadata):
    """Metadata for a processed image with detected regions."""

    core: CoreSegmentResult | None = None
    tray: ImageSegmentResult | None = None
    ruler: RulerSegmentResult | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: ImageMetadata,
        core: CoreSegmentResult | None = None,
        tray: ImageSegmentResult | None = None,
        ruler: RulerSegmentResult | None = None,
    ) -> "ImageMetadataProcessed":
        """Construct an ImageMetadataProcessed from an existing ImageMetadata.

        Args:
            metadata (ImageMetadata): The original image metadata.
            core (CoreSegmentResult | None): Detected core bounding box, if any.
            tray (ImageSegmentResult | None): Detected tray bounding box, if any.
            ruler (RulerSegmentResult | None): Detected ruler bounding box, if any.

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

    def to_dict(self) -> dict:
        """Return this processed image's metadata and detections as a plain dict, keyed by region."""
        return {
            **super().to_dict(),
            "core": self.core.to_dict() if self.core else {},
            "ruler": self.ruler.to_dict() if self.ruler else {},
            "tray": self.tray.to_dict() if self.tray else {},
        }
