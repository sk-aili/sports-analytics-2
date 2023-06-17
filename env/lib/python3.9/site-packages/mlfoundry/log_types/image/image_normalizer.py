# The logic is copied from
# https://github.com/truefoundry/mlfoundry-server/blob/c2334ace8f35322cd5e50c275b8df08327688c01/mlflow/tracking/client.py#L1188
from typing import Union

import numpy as np

from mlfoundry.exceptions import MlFoundryException
from mlfoundry.log_types.image.constants import MISSING_PILLOW_PACKAGE_MESSAGE
from mlfoundry.logger import logger
from mlfoundry.run_utils import get_module


def _normalize_to_uint8(x):
    # Based on: https://github.com/matplotlib/matplotlib/blob/06567e021f21be046b6d6dcf00380c1cb9adaf3c/lib/matplotlib/image.py#L684

    is_int = np.issubdtype(x.dtype, np.integer)
    low = 0
    high = 255 if is_int else 1
    if x.min() < low or x.max() > high:
        msg = (
            "Out-of-range values are detected. "
            "Clipping array (dtype: '{}') to [{}, {}]".format(x.dtype, low, high)
        )
        logger.warning(msg)
        x = np.clip(x, low, high)

    # float or bool
    if not is_int:
        x = x * 255

    return x.astype(np.uint8)


def _convert_numpy_to_pil_image(image: np.ndarray) -> "PIL.Image.Image":
    pil_image_module = get_module(
        module_name="PIL.Image",
        required=True,
        error_message=MISSING_PILLOW_PACKAGE_MESSAGE,
    )
    valid_data_types = {
        "b": "bool",
        "i": "signed integer",
        "u": "unsigned integer",
        "f": "floating",
    }

    if image.dtype.kind not in valid_data_types.keys():
        raise TypeError(
            "Invalid array data type: '{}'. Must be one of {}".format(
                image.dtype, list(valid_data_types.values())
            )
        )

    if image.ndim not in [2, 3]:
        raise ValueError(
            "`image` must be a 2D or 3D array but got a {}D array".format(image.ndim)
        )

    if (image.ndim == 3) and (image.shape[2] not in [1, 3, 4]):
        raise ValueError(
            "Invalid channel length: {}. Must be one of [1, 3, 4]".format(
                image.shape[2]
            )
        )

    # squeeze a 3D grayscale image since `Image.fromarray` doesn't accept it.
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    image = _normalize_to_uint8(image)

    return pil_image_module.fromarray(image)


def normalize_image(
    image: Union["PIL.Image.Image", "numpy.ndarray"]
) -> "PIL.Image.Image":
    pil_image_module = get_module(
        module_name="PIL.Image",
        required=True,
        error_message=MISSING_PILLOW_PACKAGE_MESSAGE,
    )
    if isinstance(image, pil_image_module.Image):
        return image
    if isinstance(image, np.ndarray):
        return _convert_numpy_to_pil_image(image)
    raise MlFoundryException(
        f"image should be of type PIL Image/np.ndarray got type {type(image)}"
    )
