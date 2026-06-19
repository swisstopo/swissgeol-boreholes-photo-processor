"""Module for image segmentation."""

from PIL import Image
from tqdm import tqdm

from src.mlflow_utils import log_artifact_with_mlflow
from src.models import CoreSegmentResult, ImageMetadata, ImageMetadataProcessed


def segment(imgs_metadata: list[ImageMetadata], with_mlflow: bool = False) -> list[ImageMetadataProcessed]:
    """Segment the input images and return a list of detections.

    Args:
        imgs_metadata (list[ImageMetadata]): A list of image metadata objects to be segmented.
        with_mlflow (bool): Whether to log artifacts to MLflow.

    Returns:
        list[ImageMetadataProcessed]: A list of processed image metadata objects, one per input image.
    """
    detections: list[ImageMetadataProcessed] = []

    for img_metadata in tqdm(imgs_metadata, desc="Segmenting images"):
        with Image.open(img_metadata.image_path) as img:
            detection = img.copy()  # placeholder
            w, h = img.size

        bounding_box = (0.0, h * 2 / 3, float(w), float(h))  # placeholder: lower third of the image

        if with_mlflow:
            log_artifact_with_mlflow(
                img=detection,
                filename=f"{img_metadata.image_path.stem}",
                bounding_box=bounding_box,
            )

        detections.append(
            ImageMetadataProcessed.from_metadata(
                metadata=img_metadata,
                result=CoreSegmentResult(bounding_box=bounding_box),
            )
        )

    return detections
