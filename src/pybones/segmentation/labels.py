"""Label values stored in label images and volumes."""

from enum import IntEnum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Iterable


class Label(IntEnum):
    """Voxel classes.

    ``AMBIGUOUS`` only exists in intermediate volumes: small bone-intensity blobs
    that cannot be told apart from dim (partially filled) vessels in 2D. The 3D
    refinement resolves them; final volumes contain the other four labels only.
    """

    BACKGROUND = 0
    BONE = 1
    VESSEL = 2
    BURST = 3
    AMBIGUOUS = 4

    @property
    def key(self) -> str:
        """Lower-case name used in configuration files and on the command line."""
        return self.name.lower()

    @classmethod
    def from_key(cls, key: str) -> Self:
        """Parse a label from its :attr:`key`.

        Args:
            key: Label key, e.g. ``"vessel"`` (case-insensitive).

        Returns:
            The label.

        Raises:
            ValueError: If ``key`` names no label.
        """
        try:
            return cls[key.upper()]
        except KeyError:
            raise ValueError(f"Unknown label {key!r}; expected one of {label_keys()}") from None


#: Labels present in final (refined) volumes.
FINAL_LABELS: tuple[Label, ...] = (Label.BACKGROUND, Label.BONE, Label.VESSEL, Label.BURST)

#: Labels that can be meshed or exported (everything but the background).
FOREGROUND_LABELS: tuple[Label, ...] = (Label.BONE, Label.VESSEL, Label.BURST)


def label_keys(labels: "Iterable[Label]" = Label) -> list[str]:
    """List the keys of ``labels``.

    Args:
        labels: Labels to name; the :class:`Label` enum itself (the default) iterates
            over every label.

    Returns:
        The :attr:`Label.key` of each label, in order.
    """
    return [lab.key for lab in labels]
