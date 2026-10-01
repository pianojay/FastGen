from typing import Callable, Optional

from data.parquet_imagenet import ImagenetParquetDataset


def create_dataset(
    image_transform: Optional[Callable] = None,
    **kwargs,
):
    # Feel free to replace the following to your dataloading class.
    
    return ImagenetParquetDataset(
        path="./imagenet",
        image_transform=image_transform,
        **kwargs,
    )
