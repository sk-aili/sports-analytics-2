import re

IMAGE_METADATA_FILE_NAME = "image_metadata.json"
DEFAULT_IMAGE_FORMAT = "png"
MISSING_PILLOW_PACKAGE_MESSAGE = (
    "We need PIL package to save image.\nTo install, run `pip install pillow`"
)
IMAGE_KEY_REGEX = re.compile(r"^[a-zA-Z0-9-_]+$")
