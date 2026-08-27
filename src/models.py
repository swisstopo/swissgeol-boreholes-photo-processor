"""Module containing data models for the borehole image processing application."""

import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum, auto
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from src.utils import get_image_shape, load_image


@dataclass
class ImageMetadata:
    """Shared metadata for an image file, common to both the cores and cuttings pipelines.

    borehole_id is the filename prefix identifying the borehole; image_path is the source file.
    """

    borehole_id: str
    image_path: Path

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
        """Load an image and normalize it to an RGB float array in [0, 1].

        Args:
            factor (float): Downscale factor applied after loading; 1.0 leaves the image unscaled.

        Returns:
            np.ndarray: RGB image array with float values in [0, 1].

        Raises:
            SegmentationError: If the image is not a 3/4-channel array.
            ValueError: If the image has an unsupported dtype.
        """
        return load_image(str(self.image_path), factor)

    def to_dict(self) -> dict:
        """Return this metadata as a plain dict, e.g. for JSON serialization."""
        return {"borehole_id": self.borehole_id, "image_path": self.image_path}


@dataclass
class ImageMetadataCores(ImageMetadata):
    """Metadata for a core image file.

    borehole_id is the filename prefix before the depth range.
    depth_start and depth_end are parsed from the filename, e.g.
    ``GBC-CB50_0015.00-0016.00_vd_p.TIF``.
    """

    depth_start: float
    depth_end: float

    _DEPTH_PATTERN: ClassVar[re.Pattern] = re.compile(r"_(?P<depth_start>\d+\.\d+)-(?P<depth_end>\d+\.\d+)")

    @classmethod
    def from_path(cls, image_path: Path) -> "ImageMetadataCores":
        """Construct an ImageMetadataCores from an image path.

        borehole_id is extracted as the filename prefix before the depth range.
        depth_start and depth_end are extracted from the filename via regex.

        Args:
            image_path (Path): Full path to an image file, e.g.
                ``Path(".../GBC/GBC-CB50/GBC-CB50_0015.00-0016.00_vd_p.TIF")``.

        Raises:
            ValueError: If no depth range can be found in the filename, or if the
                parsed depth_end is not strictly greater than depth_start.

        Returns:
            ImageMetadataCores: An instance containing the parsed metadata.
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


@dataclass
class ImageMetadataCuttings(ImageMetadata):
    """Metadata for a cuttings image file.

    borehole_id is assigned by the caller from the input folder name, since cuttings
    filenames follow several borehole-specific conventions and don't reliably carry a
    usable id prefix (see collect_cuttings). depth is a single point depth parsed from
    the filename, e.g. ``GES-F-1 190 m (Large).JPG``.
    """

    depth: float

    # Forsthaus, e.g. "GES-F-1 190 m (Large).JPG": id prefix, then a single point depth.
    # The depth must start at whitespace/string-start so it can't match a digit embedded
    # in the id itself (e.g. the "1" in "GES-F-1"); a trailing annotation is either
    # parenthesized or a single letters-only word (no digits, so it can't mask a number).
    _DEPTH_REGEX_FORSTHAUS: ClassVar[re.Pattern] = re.compile(
        r"(?:^|(?<=\s))(?P<depth>\d+(?:\.\d+)?)[\s.]*m?(?:\s*\(.*\)|\s+[A-Za-z]+)?$", re.IGNORECASE
    )

    # iOS Photos export names, e.g. "6C296742-39EF-4423-97F0-B7428B70B5CE_1_105_c.jpeg": a
    # UUID can start with a hex digit (0-9), which would otherwise be misread by
    # _DEPTH_REGEX_PLAIN below as a one-digit depth. Rejected outright before that regex
    # gets a chance at it; these carry no depth anywhere in the name.
    _UUID_REGEX: ClassVar[re.Pattern] = re.compile(
        r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}", re.IGNORECASE
    )

    # GEo-01 and GVL-1, e.g. "84m_00.JPG", "536-538m_00.JPG", "1450b_IMG_20240609_135045.jpg"
    # or "2100 to 2130m.jpeg": no id prefix, filename starts directly with the depth
    # (optionally a range using "-" or "to", in which case the range end is used) and an
    # optional "m" unit; everything after that (version letters/digits glued directly,
    # underscore- or space-separated camera/annotation text, "(2)" copy suffixes, ...) is
    # discarded wholesale, since there's no id prefix here to risk matching instead.
    _DEPTH_REGEX_PLAIN: ClassVar[re.Pattern] = re.compile(
        r"^(?P<depth>\d+(?:\.\d+)?)(?:\s*(?:-|to)\s*(?P<depth_end>\d+(?:\.\d+)?))?m?.*$", re.IGNORECASE
    )

    # Vinzel-1-Malm, e.g. "V1SM_1000m_02.jpg" or "V1SM_1807.5m_02.jpg": fixed "V1SM_" prefix,
    # then the depth, an optional "m" unit, and a sequence/annotation suffix discarded
    # wholesale; same shape as _DEPTH_REGEX_PLAIN but anchored past the fixed prefix instead
    # of directly at the depth, since here the depth isn't the first token in the filename.
    _DEPTH_REGEX_V1SM: ClassVar[re.Pattern] = re.compile(r"^V1SM_(?P<depth>\d+(?:\.\d+)?)m?.*$", re.IGNORECASE)

    # GVL-1, e.g. "IMG_20240604_160011_610.jpg" or "IMG_20240526_205320_475m.jpg": a plain
    # camera filename (date + time) with the depth appended as a trailing token instead of
    # leading. Camera files with no depth anywhere (e.g. "IMG_20240611_135741.jpg") don't
    # match this and are left to fail as unparseable.
    _DEPTH_REGEX_IMG_TRAILING: ClassVar[re.Pattern] = re.compile(
        r"^IMG_\d{8}_\d{6}_(?P<depth>\d+(?:\.\d+)?)m?$", re.IGNORECASE
    )

    # GEo-02, e.g. "GEo02_1014-1018-1.JPG" or "GEo02_038-1.JPG": after the "GEo<n>_"
    # prefix, numbers are separated by dashes/annotations with no consistent grammar, so
    # rather than one regex we just pull out every number and reason about magnitude: a
    # small second number is a version index (every real version in the data is <= 6),
    # otherwise it's a range end and becomes the depth; any third number is always a
    # version index. Verified against all 1655 real GEo-02 filenames with zero exceptions.
    _GEO_PREFIX_REGEX: ClassVar[re.Pattern] = re.compile(r"^GEo\d+_", re.IGNORECASE)
    _NUMBER_REGEX: ClassVar[re.Pattern] = re.compile(r"\d+(?:\.\d+)?")
    _GEO_VERSION_MAX: ClassVar[float] = 10.0

    # Montagny-2/-2ST, e.g. "MONTAGNY-2_Cuttings_0060.00-0065.00.jpg" or
    # "MONTAGNY-2ST_Cuttings_1059.00-1060.00.jpeg": id prefix, then an unambiguous depth
    # range (both numbers decimal), same shape as the ImageMetadataCores convention; the
    # range end is used, matching the range rule used elsewhere in this class.
    _DEPTH_REGEX_RANGE: ClassVar[re.Pattern] = re.compile(r"_(?P<depth_start>\d+\.\d+)-(?P<depth_end>\d+\.\d+)")

    @classmethod
    def from_path(cls, image_path: Path) -> "ImageMetadataCuttings":
        """Construct an ImageMetadataCuttings from an image path.

        depth is extracted from the filename; the GEo-02, IMG-trailing-depth, plain
        (GEo-01/GVL-1), Vinzel-1-Malm, Montagny-range and Forsthaus naming conventions
        are tried in turn (see the regexes above for each format's shape). borehole_id
        is left blank; the caller assigns it from the input folder name.

        Args:
            image_path (Path): Full path to an image file, e.g.
                ``Path(".../GES-F-1/GES-F-1 190 m (Large).JPG")``.

        Raises:
            ValueError: If no depth can be found in the filename under any convention.

        Returns:
            ImageMetadataCuttings: An instance containing the parsed metadata.
        """
        stem = image_path.stem

        if cls._UUID_REGEX.match(stem):
            raise ValueError(f"No depth found in filename: {image_path.name}")

        geo_prefix_match = cls._GEO_PREFIX_REGEX.match(stem)
        if geo_prefix_match:
            numbers = cls._NUMBER_REGEX.findall(stem[geo_prefix_match.end() :])
            if not numbers:
                raise ValueError(f"No depth found in filename: {image_path.name}")
            depth = float(numbers[0])
            if len(numbers) > 1 and float(numbers[1]) >= cls._GEO_VERSION_MAX:
                depth = float(numbers[1])
            return cls(borehole_id="", depth=depth, image_path=image_path)

        img_trailing_match = cls._DEPTH_REGEX_IMG_TRAILING.match(stem)
        if img_trailing_match:
            depth = float(img_trailing_match.group("depth"))
            return cls(borehole_id="", depth=depth, image_path=image_path)

        plain_match = cls._DEPTH_REGEX_PLAIN.match(stem)
        if plain_match:
            depth = float(plain_match.group("depth_end") or plain_match.group("depth"))
            return cls(borehole_id="", depth=depth, image_path=image_path)

        v1sm_match = cls._DEPTH_REGEX_V1SM.match(stem)
        if v1sm_match:
            depth = float(v1sm_match.group("depth"))
            return cls(borehole_id="", depth=depth, image_path=image_path)

        range_match = cls._DEPTH_REGEX_RANGE.search(stem)
        if range_match:
            depth = float(range_match.group("depth_end"))
            return cls(borehole_id="", depth=depth, image_path=image_path)

        match = cls._DEPTH_REGEX_FORSTHAUS.search(stem)
        if not match:
            raise ValueError(f"No depth found in filename: {image_path.name}")
        depth = float(match.group("depth"))
        return cls(borehole_id="", depth=depth, image_path=image_path)

    def to_dict(self) -> dict:
        """Return this metadata as a plain dict, e.g. for JSON serialization."""
        return {**super().to_dict(), "depth": self.depth}


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
    def approach_to_json(results: list["ImageSegmentResult | None"]) -> dict[str, int | float]:
        """Summarize per-approach detection counts and average timing across a batch.

        Args:
            results (list[ImageSegmentResult | None]): Per-image detection results for one region
                (e.g. every tray detection across a batch). None entries mark images where
                detection failed.

        Returns:
            dict[str, int | float]: Counts of images that failed (n_as_fail), used the per-image
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
            "time_group_total_avg": sum(ts_group) / (len(ts_group) + 1e-16),
        }

    def to_dict(self) -> dict:
        """Return this result as a plain dict, e.g. for JSON serialization."""
        return {"approach": self.approach.name, "bbox": self.bbox, "time": self.time}


@dataclass
class CoreSegmentResult(ImageSegmentResult):
    """Result of detecting the core bbox (bbox is used downstream for cropping/evaluation/stitching)."""

    # Per-segment bboxes before merging into `bbox`; kept only for MLflow debug visualization.
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
class ImageMetadataProcessedCores(ImageMetadataCores):
    """Metadata for a processed image with detected regions."""

    core: CoreSegmentResult | None = None
    tray: ImageSegmentResult | None = None
    ruler: RulerSegmentResult | None = None
    _core_cache: Image.Image | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: ImageMetadataCores,
        core: CoreSegmentResult | None = None,
        tray: ImageSegmentResult | None = None,
        ruler: RulerSegmentResult | None = None,
        preload: bool = False,
    ) -> "ImageMetadataProcessedCores":
        """Construct an ImageMetadataProcessedCores from an existing ImageMetadata.

        Args:
            metadata (ImageMetadataCores): The original image metadata.
            core (CoreSegmentResult | None): Detected core bounding box, if any.
            tray (ImageSegmentResult | None): Detected tray bounding box, if any.
            ruler (RulerSegmentResult | None): Detected ruler bounding box, if any.
            preload (bool): If True, eagerly crop and cache the core image via `load_core()`
                right away instead of deferring to first access.

        Returns:
            ImageMetadataProcessedCores: A new instance containing the original metadata and the processing result.
        """
        obj = cls(
            borehole_id=metadata.borehole_id,
            depth_start=metadata.depth_start,
            depth_end=metadata.depth_end,
            image_path=metadata.image_path,
            core=core,
            tray=tray,
            ruler=ruler,
        )
        if preload:
            obj.load_core()

        return obj

    def load_core(self) -> Image.Image:
        """Cut a core segment from the source image, rotating to portrait if needed.

        Reuses load_image for the source pixels, which already rotates portrait images to
        landscape to match the coordinate space self.core.bbox was detected in. Cores are
        stored vertically in the output, so landscape crops (width > height) are rotated 90°
        clockwise so the left edge (shallow end) becomes the top. The crop is cached after
        first access so repeated calls (e.g. during parallel stitching) don't re-read the
        source file.

        Returns:
            Image.Image: The cropped core segment image in portrait orientation.

        Raises:
            ValueError: If no core region was detected for this image.
        """
        if self.core is None:
            raise ValueError(f"No core region detected for image: {self.image_path}")

        if self._core_cache is None:
            src = self.load_image()
            left, upper, right, lower = (round(v) for v in self.core.bbox)
            crop = Image.fromarray((255 * src[upper:lower, left:right]).astype(np.uint8))
            if crop.width > crop.height:
                crop = crop.transpose(Image.Transpose.ROTATE_270)  # clockwise: left (shallow) → top
            self._core_cache = crop

        return self._core_cache

    def to_dict(self) -> dict:
        """Return this processed image's metadata and detections as a plain dict, keyed by region."""
        return {
            **super().to_dict(),
            "core": self.core.to_dict() if self.core else {},
            "ruler": self.ruler.to_dict() if self.ruler else {},
            "tray": self.tray.to_dict() if self.tray else {},
        }


class PaperDetectionStatus(StrEnum):
    """Outcome of the pebble method's reference-paper-sheet detection (segment_pebble).

    Unset (None) for cuttings methods that don't rely on paper detection (e.g. black_circle).
    """

    FOUND = "found"
    NO_CANDIDATE = "no_candidate"  # no region passed the shape/area/edge-anchoring filters, at either threshold
    DEGENERATE_LEFT_EDGE = "degenerate_left_edge"  # detected paper's left edge is at column 0 -- nothing to crop
    CROPPED_TOO_MUCH = "cropped_too_much"  # detected paper would crop away more than max_cropped_frac of the image


@dataclass
class CuttingsSegmentResult(ImageSegmentResult):
    """Result of detecting the cuttings bbox (bbox is used downstream for cropping/evaluation/stitching)."""

    # Per-segment bboxes before merging into `bbox`; kept only for MLflow debug visualization.
    bbox_segments: list[tuple[float, float, float, float]] | None = None
    paper_status: PaperDetectionStatus | None = None  # outcome of paper detection; see PaperDetectionStatus

    def to_dict(self) -> dict:
        """Return this result as a plain dict, including the paper detection status."""
        return {**super().to_dict(), "paper_status": self.paper_status.value if self.paper_status else None}

    @staticmethod
    def paper_status_counts(results: list["CuttingsSegmentResult | None"]) -> dict[str, int]:
        """Count how many results ended in each PaperDetectionStatus outcome.

        Only counts results that actually set a paper_status (i.e. went through segment_pebble);
        results from other cuttings methods, or failed detections (None), are ignored.

        Args:
            results (list[CuttingsSegmentResult | None]): Per-image cuttings detection results for one batch.

        Returns:
            dict[str, int]: Number of results per PaperDetectionStatus value.
        """
        statuses = [result.paper_status for result in results if result and result.paper_status is not None]
        return {status.value: statuses.count(status) for status in PaperDetectionStatus}


@dataclass
class ImageMetadataProcessedCuttings(ImageMetadataCuttings):
    """Metadata for a processed image with detected regions."""

    cuttings: CuttingsSegmentResult | None = None
    _cuttings_cache: Image.Image | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: ImageMetadataCuttings,
        cuttings: CuttingsSegmentResult | None = None,
        preload: bool = False,
    ) -> "ImageMetadataProcessedCuttings":
        """Construct an ImageMetadataProcessedCuttings from an existing ImageMetadata.

        Args:
            metadata (ImageMetadataCuttings): The original image metadata.
            cuttings (CuttingsSegmentResult | None): Detected cuttings bounding box, if any.
            preload (bool): If True, eagerly load and cache the cutting image right away.

        Returns:
            ImageMetadataProcessedCuttings: A new instance containing the original metadata and the processing result.
        """
        obj = cls(
            borehole_id=metadata.borehole_id,
            depth=metadata.depth,
            image_path=metadata.image_path,
            cuttings=cuttings,
        )
        if preload:
            obj.load_cuttings()

        return obj

    def load_cuttings(self) -> Image.Image:
        """Cut the cuttings segment from the source image.

        Reuses load_image for the source pixels, which already rotates portrait images to
        landscape to match the coordinate space self.cuttings.bbox was detected in. The crop
        is cached after first access so repeated calls (e.g. during parallel stitching) don't
        re-read the source file.

        Returns:
            Image.Image: The cropped cuttings segment image, in landscape orientation.

        Raises:
            ValueError: If no cuttings region was detected for this image.
        """
        if self.cuttings is None:
            raise ValueError(f"No cuttings region detected for image: {self.image_path}")

        if self._cuttings_cache is None:
            src = self.load_image()
            left, upper, right, lower = (round(v) for v in self.cuttings.bbox)
            self._cuttings_cache = Image.fromarray((255 * src[upper:lower, left:right]).astype(np.uint8))

        return self._cuttings_cache

    def to_dict(self) -> dict:
        """Return this processed image's metadata and detections as a plain dict, keyed by region."""
        return {
            **super().to_dict(),
            "cuttings": self.cuttings.to_dict() if self.cuttings else {},
        }
