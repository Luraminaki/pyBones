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
#: Micrometres in a millimetre (lengths are µm inside, reported in mm).
UM_PER_MM = 1000.0
#: Binary units.
KIB_PER_MIB = 1024
MIB_PER_GIB = 1024
#: Suffix of a file being written; it is renamed to its final name once complete, so that a
#: reader never sees a partial file and an interrupted write leaves nothing behind.
PARTIAL_SUFFIX = ".partial"
