"""Format facts shared across the package and its tools (not tunable: see the configuration)."""

#: 8-bit images: largest value and number of levels.
UINT8_MAX = 255
UINT8_LEVELS = UINT8_MAX + 1
#: 16-bit images (µCT slices): largest value and number of levels.
UINT16_MAX = 65535
UINT16_LEVELS = UINT16_MAX + 1
#: Colour images are BGR (OpenCV's order).
BGR_CHANNELS = 3
#: Indentation of the JSON files written by pyBones and its tools.
JSON_INDENT = 2
#: Binary units.
KIB_PER_MIB = 1024
MIB_PER_GIB = 1024
