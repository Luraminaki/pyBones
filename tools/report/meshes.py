"""Mesh files of today's segmentation: sizes, counts, position error and timings.

:func:`mesh_files` meshes each foreground label of the reference run and writes it in every
format the report compares (STL, 32-bit GLB, 16-bit GLB) into a temporary folder, then
records what was measured in ``meshes.csv``. :func:`mesh_values` turns that file into the
report's macros, and :func:`readme_meshes` into the README's table, so that no mesh
figure is typed by hand.
"""

import csv
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pybones.meshing.gltf import FLOAT_POSITION_BITS, QUANTIZED_POSITION_BITS, read_glb, write_glb
from pybones.meshing.marching import extract_surface
from pybones.meshing.stl import write_stl
from pybones.segmentation.labels import FOREGROUND_LABELS, Label

if TYPE_CHECKING:
    from pybones.meshing.mesh import Mesh
    from tools.report.context import Context

MESHES_CSV = "meshes.csv"
README_MESHES_START = "<!-- meshes:start (written by python -m tools.report figures) -->"
README_MESHES_END = "<!-- meshes:end -->"
#: The label whose files the report and the README compare in every format.
COMPARED_LABEL = Label.VESSEL
_COLUMNS: tuple[str, ...] = (
    "label",
    "variant",
    "bytes",
    "points",
    "triangles",
    "max_error_um",
    "mesh_seconds",
)
_MEGA = 1e6
_STL_HEADER = "pybones report"


class Variant(StrEnum):
    """File formats compared (STL, and GLB at both position precisions)."""

    STL = "stl"
    GLB_FLOAT = "glb32"
    GLB_QUANTIZED = "glb16"

    @property
    def heading(self) -> str:
        """Name in the tables."""
        return {
            Variant.STL: "STL",
            Variant.GLB_FLOAT: "GLB, 32-bit",
            Variant.GLB_QUANTIZED: "GLB, 16-bit",
        }[self]

    @property
    def use(self) -> str:
        """What the format is for, in the tables."""
        return {
            Variant.STL: "other software (ImageJ, MeshLab)",
            Variant.GLB_FLOAT: "browser, exact positions",
            Variant.GLB_QUANTIZED: "browser (default)",
        }[self]


def _write(mesh: "Mesh", variant: Variant, path: Path, name: str, chunk_faces: int) -> float:
    """Write a mesh in one variant and measure its largest position error.

    Args:
        mesh: Mesh in µm.
        variant: File format.
        path: Destination file.
        name: Mesh name (the label key).
        chunk_faces: Triangles processed at a time.

    Returns:
        The largest distance, along any axis, between a stored and a true position (µm).
    """
    if variant == Variant.STL:
        _ = write_stl(mesh, path, _STL_HEADER, chunk_faces)
        return 0.0
    bits = FLOAT_POSITION_BITS if variant == Variant.GLB_FLOAT else QUANTIZED_POSITION_BITS
    _ = write_glb(mesh, path, name, bits, chunk_faces)
    stored = read_glb(path).vertices
    return float(np.abs(stored - mesh.vertices.astype(np.float64)).max())


def mesh_files(ctx: "Context") -> None:
    """Mesh every foreground label of today's segmentation and measure its files.

    Args:
        ctx: Figure context.
    """
    labels = ctx.reference_labels()
    voxel = ctx.config.acquisition.voxel_size_um
    chunk = ctx.config.mesh.chunk_faces
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(dir=ctx.data) as scratch:
        for label in FOREGROUND_LABELS:
            start = time.perf_counter()
            mesh = extract_surface(labels, label, ctx.config.mesh, voxel)
            seconds = time.perf_counter() - start
            for variant in Variant:
                path = Path(scratch) / f"{label.key}.{variant.value}"
                error = _write(mesh, variant, path, label.key, chunk)
                rows.append(
                    {
                        "label": label.key,
                        "variant": variant.value,
                        "bytes": path.stat().st_size,
                        "points": len(mesh.vertices),
                        "triangles": mesh.n_faces,
                        "max_error_um": f"{error:.6f}",
                        "mesh_seconds": f"{seconds:.1f}",
                    }
                )
                path.unlink()
    with (ctx.data / MESHES_CSV).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rows(ctx: "Context") -> dict[tuple[str, str], dict[str, str]]:
    """Read ``meshes.csv``.

    Args:
        ctx: Figure context.

    Returns:
        ``(label key, variant)`` -> row (column -> text).
    """
    with (ctx.data / MESHES_CSV).open() as fh:
        return {(row["label"], row["variant"]): row for row in csv.DictReader(fh)}


def mesh_values(ctx: "Context") -> dict[str, str]:
    """Turn the measured mesh files into report macros (names: letters only).

    Args:
        ctx: Figure context.

    Returns:
        Macro name (without ``pbValue``) -> formatted value.
    """
    rows = _rows(ctx)
    values: dict[str, str] = {}
    for label in FOREGROUND_LABELS:
        name = label.key.capitalize()
        quantized = rows[(label.key, Variant.GLB_QUANTIZED.value)]
        values[f"Mesh{name}Triangles"] = f"{int(quantized['triangles']) / _MEGA:.2f}"
        values[f"Mesh{name}Points"] = f"{int(quantized['points']) / _MEGA:.2f}"
        values[f"Mesh{name}Seconds"] = f"{float(quantized['mesh_seconds']):.0f}"
        values[f"Mesh{name}GlbMB"] = f"{int(quantized['bytes']) / _MEGA:.1f}"
    compared = COMPARED_LABEL.key
    values["MeshComparedStlMB"] = f"{int(rows[(compared, Variant.STL.value)]['bytes']) / _MEGA:.1f}"
    values["MeshComparedGlbFloatMB"] = (
        f"{int(rows[(compared, Variant.GLB_FLOAT.value)]['bytes']) / _MEGA:.1f}"
    )
    error = float(rows[(compared, Variant.GLB_QUANTIZED.value)]["max_error_um"])
    voxel = min(ctx.config.acquisition.voxel_size_um)
    values["MeshErrorUm"] = f"{error:.2f}"
    values["MeshErrorVoxelRatio"] = f"{voxel / error:.0f}" if error > 0 else "inf"
    grid = ctx.config.mesh.factor_for(COMPARED_LABEL) * voxel
    values["MeshGridUm"] = f"{grid:g}"
    return values


def readme_meshes(ctx: "Context") -> None:
    """Write the mesh file table into the README, between the mesh markers.

    Args:
        ctx: Figure context.

    Raises:
        ValueError: If the README lacks the markers.
    """
    path = ctx.repo / ctx.settings.readme
    text = path.read_text()
    start, end = text.find(README_MESHES_START), text.find(README_MESHES_END)
    if start < 0 or end < start:
        raise ValueError(
            f"{path} lacks the {README_MESHES_START!r} ... {README_MESHES_END!r} markers"
        )
    rows = _rows(ctx)
    compared = COMPARED_LABEL.key
    first = rows[(compared, Variant.GLB_QUANTIZED.value)]
    lines = [
        f"The {compared} mesh of the sample ({int(first['points']) / _MEGA:.2f} million points, "
        + f"{int(first['triangles']) / _MEGA:.2f} million triangles), with pyBones {ctx.version}:",
        "",
        "| Format | Size | Largest position error | Use |",
        "|---|---|---|---|",
    ]
    for variant in Variant:
        row = rows[(compared, variant.value)]
        error = float(row["max_error_um"])
        shown = "none (32-bit floats)" if error == 0 else f"{error:.2f} µm"
        lines.append(
            f"| {variant.heading} | {int(row['bytes']) / _MEGA:.1f} MB | {shown} | {variant.use} |"
        )
    table = "\n".join([README_MESHES_START, *lines, README_MESHES_END])
    _ = path.write_text(text[:start] + table + text[end + len(README_MESHES_END) :])
