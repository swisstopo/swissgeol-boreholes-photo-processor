"""Shared helpers used across the segmentation utils package."""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from timeit import default_timer as timer
from typing import Generic, TypeVar

import numpy as np

from src.models import ApproachType, ImageMetadataCores, ImageSegmentResult

K = TypeVar("K", bound=ImageSegmentResult)
T = TypeVar("T")

logger = logging.getLogger(__name__)


def group_images_by_shape(
    imgs_metadata: list[ImageMetadataCores],
) -> dict[tuple[int, int, int], list[ImageMetadataCores]]:
    """Group images by their shape (height, width, channels).

    Args:
        imgs_metadata (list[ImageMetadataCores]): List of image metadata.

    Returns:
        dict[tuple[int, int, int], list[ImageMetadataCores]]: Dictionary where keys are image shapes
            and values are lists of ImageMetadataCores with that shape.
    """
    grouped: dict[tuple[int, int, int], list[ImageMetadataCores]] = {}
    for img_metadata in imgs_metadata:
        grouped.setdefault(img_metadata.shape, []).append(img_metadata)
    return grouped


class ProcessGroupByShape(ABC, Generic[K, T]):
    """Abstract base class for processing groups of images into a single aggregated result.

    Images are grouped by shape (see `group_images_by_shape`), a fixed-size random sample is
    drawn per group, and each sampled image is preprocessed in parallel via `_preprocess` before
    being combined by `_aggregate` into a per-shape result.

    Type Parameters:
        K: Type of the aggregated result returned per shape group.
        T: Type of the per-image value returned by `_preprocess`.
    """

    def __init__(
        self,
        min_group_size: int = 10,
        seed: int = 0,
        n_workers: int = 1,
    ):
        """Configure the sampling and parallelism used when processing shape groups.

        Args:
            min_group_size (int, optional): Minimum number of images required in a shape group
                for it to be processed; smaller groups are skipped. Also the number of images
                sampled per group. Defaults to 10.
            seed (int, optional): Seed for the random sampling of images within each group.
                Defaults to 0.
            n_workers (int, optional): Number of worker processes used to run `_preprocess` in
                parallel. Defaults to 1
        """
        super().__init__()

        self.min_group_size = min_group_size
        self.n_workers = n_workers
        self.seed = seed

    @abstractmethod
    def _preprocess(self, img_metadata: ImageMetadataCores, img_metadata_ref: ImageMetadataCores) -> T | None:
        """Preprocess a single image ahead of aggregation.

        Args:
            img_metadata (ImageMetadataCores): Metadata of the image to preprocess.
            img_metadata_ref (ImageMetadataCores): Reference image for image processing.

        Returns:
            T | None: The preprocessed value to feed into `_aggregate`, or None.
        """
        ...

    @abstractmethod
    def _aggregate(self, processed_items: list[T]) -> K | None:
        """Combine the preprocessed items of one shape group into a single result.

        Args:
            processed_items (list[T]): Preprocessed values returned by `_preprocess` for a
                shape group.

        Returns:
            K | None: The aggregated result for the group, or None.
        """
        ...

    def _run_group(self, executor: ProcessPoolExecutor, imgs_metadata: list[ImageMetadataCores]) -> K | None:
        """Preprocess every image in a shape group (in parallel) and aggregate the results.

        Args:
            executor (ProcessPoolExecutor): Pool shared across all shape groups in `run`.
            imgs_metadata (list[ImageMetadataCores]): Images to preprocess and aggregate.

        Returns:
            K | None: The aggregated result for the group, or None.
        """
        processed_items = list(
            executor.map(partial(self._preprocess, img_metadata_ref=imgs_metadata[0]), imgs_metadata)
        )

        # Remove unwanted detections
        processed_items = [item for item in processed_items if item is not None]

        return self._aggregate(processed_items)

    def run(self, imgs_metadata: list[ImageMetadataCores]) -> dict[tuple[int, int, int], K]:
        """Group images by shape, sample, and aggregate a result per shape.

        Args:
            imgs_metadata (list[ImageMetadataCores]): Images to group and process.

        Returns:
            dict[tuple[int, int, int], K]: Aggregated result per image shape.
        """
        rng = np.random.default_rng(self.seed)

        groups = group_images_by_shape(imgs_metadata)
        groups = {key: values for key, values in groups.items() if len(values) >= self.min_group_size}

        results = {}

        # Reuse a single pool across all shape groups instead of paying its startup cost per group
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            for i, (shape, group) in enumerate(groups.items()):
                logger.info(f"[{i + 1}/{len(groups)}] Extracting group {shape} with {len(group)} samples")

                # A fixed-size sample per group is selected for estimation and aggregation
                sample_ids = rng.choice(len(group), size=self.min_group_size, replace=False)

                # Measure execution time
                t_start = timer()
                result = self._run_group(executor, [group[i] for i in sample_ids])

                if result is not None:
                    result.time = timer() - t_start
                    result.approach = ApproachType.GROUP
                    results[shape] = result

        return results
