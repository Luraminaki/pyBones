"""``python -m tools.annotation {prepare,serve,score}`` (run from the repository root)."""

import argparse
import logging
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from pydantic import ValidationError

from pybones import __version__
from pybones.config.loader import load_config
from pybones.segmentation.results import load_labels, load_meta
from pybones.utils.logs import configure_logging
from tools.annotation.arguments import (
    AnnotationInvocation,
    Command,
    PrepareArgs,
    RegenerateArgs,
    ScoreArgs,
    ServeArgs,
)
from tools.annotation.scoring import score_segmentation
from tools.annotation.selection import extract_crops, select_crops
from tools.annotation.server import AnnotationApp, make_server
from tools.annotation.settings import DEFAULT_SETTINGS_PATH, load_settings
from tools.annotation.workspace import Manifest, Workspace, now_iso

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pybones.config.schema import PyBonesConfig
    from tools.annotation.settings import AnnotationSettings


#: Program name of the log files.
PROGRAM_NAME = "tools-annotation"


logger = logging.getLogger(__name__)


def _portable(path: Path) -> str:
    """Express ``path`` relative to the working directory (the repository) when inside it.

    Args:
        path: Path to store in the manifest.

    Returns:
        The relative path, else the absolute one.
    """
    resolved = path.resolve()
    return (
        str(resolved.relative_to(Path.cwd()))
        if resolved.is_relative_to(Path.cwd())
        else str(resolved)
    )


def cmd_prepare(
    workspace: Workspace,
    segmentation: Path,
    source: Path | None,
    force: bool,
    settings: "AnnotationSettings",
    config: "PyBonesConfig",
) -> None:
    """Choose crops from a segmentation and write their images (and pre-fills).

    Args:
        workspace: Workspace to prepare.
        segmentation: pyBones segmentation output directory.
        source: Raw stack (default: the one in the segmentation metadata).
        force: Replace an existing manifest.
        settings: Tool settings.
        config: pyBones configuration.

    Raises:
        SystemExit: If the workspace already has a manifest and ``force`` is not set.
    """
    if workspace.manifest_path.exists() and not force:
        raise SystemExit(
            f"{workspace.manifest_path} exists (painted labels would be orphaned); use --force"
        )
    meta = load_meta(segmentation, config.io)
    volume = load_labels(segmentation, config.io)
    stack = source or Path(meta.source)
    crops = select_crops(volume, settings.crops)
    if len(crops) < settings.total_crops:
        logger.warning("Only %d of %d crops qualified", len(crops), settings.total_crops)
    manifest = Manifest(
        source=_portable(stack),
        segmentation=_portable(segmentation),
        segmentation_version=meta.pybones_version,
        crop_size=settings.crops.size,
        context_slices=settings.crops.context_slices,
        prefilled=settings.prefill,
        unsure_value=settings.unsure_value,
        created=now_iso(),
        crops=crops,
    )
    workspace.save_manifest(manifest)
    extract_crops(stack, volume, crops, settings.crops, config, workspace, settings.prefill)
    logger.info("Prepared %d crops in %s", len(crops), workspace.root)


def cmd_regenerate(
    workspace: Workspace, settings: "AnnotationSettings", config: "PyBonesConfig"
) -> None:
    """Rewrite the crop images (and pre-fills) of an existing manifest, e.g. after a clone.

    Args:
        workspace: Workspace holding the manifest.
        settings: Tool settings.
        config: pyBones configuration.
    """
    manifest = workspace.load_manifest()
    volume = load_labels(Path(manifest.segmentation), config.io)
    extract_crops(
        Path(manifest.source),
        volume,
        manifest.crops,
        settings.crops.model_copy(
            update={"size": manifest.crop_size, "context_slices": manifest.context_slices}
        ),
        config,
        workspace,
        manifest.prefilled,
    )
    logger.info("Regenerated %d crops in %s", len(manifest.crops), workspace.root)


def cmd_serve(
    workspace: Workspace,
    open_browser: bool,
    settings: "AnnotationSettings",
    config: "PyBonesConfig",
) -> None:
    """Serve the annotation page until interrupted.

    Args:
        workspace: Workspace to annotate.
        open_browser: Open the page in the default browser.
        settings: Tool settings.
        config: pyBones configuration (label colours).
    """
    app = AnnotationApp(workspace, settings, config)
    server = make_server(app, settings.server.host, settings.server.port)
    host, port = server.server_address[:2]
    url = f"http://{host!s}:{port}/"
    logger.info("Annotation page: %s  (Ctrl+C to stop)", url)
    if open_browser:
        _ = webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped")
    finally:
        server.server_close()


def cmd_score(workspace: Workspace, segmentation: Path, config: "PyBonesConfig") -> None:
    """Score a segmentation against the saved annotations and log the scores.

    Args:
        workspace: Workspace holding the annotations.
        segmentation: pyBones segmentation output directory.
        config: pyBones configuration (file names).

    Raises:
        SystemExit: If no crop has been annotated yet.
    """
    manifest = workspace.load_manifest()
    score = score_segmentation(load_labels(segmentation, config.io), manifest, workspace)
    if score.crops_scored == 0:
        raise SystemExit("No annotated crop yet")
    labels = score.labels
    summary = (
        f"{score.crops_scored}/{len(manifest.crops)} crops, {labels.voxels:,} annotated px, "
        + f"accuracy {labels.accuracy:.3f} (pybones {__version__})"
    )
    logger.info("%s\n%s", summary, "\n".join(labels.report_lines("annotation")))


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        The ``python -m tools.annotation`` parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tools.annotation",
        description="Paint ground-truth labels on crops and score segmentations against them.",
    )
    _ = parser.add_argument(
        "--settings", type=Path, default=DEFAULT_SETTINGS_PATH, help="tool configuration file"
    )
    _ = parser.add_argument("--config", type=Path, help="pybones configuration file")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(Command.PREPARE, help="choose crops from a segmentation and extract them")
    _ = p.add_argument("segmentation", type=Path, help="pybones segmentation output directory")
    _ = p.add_argument("--name", required=True, help="workspace name (e.g. the dataset name)")
    _ = p.add_argument("--source", type=Path, help="raw stack (default: the one in the metadata)")
    _ = p.add_argument("--force", action="store_true", help="replace an existing manifest")

    p = sub.add_parser(Command.REGENERATE, help="rewrite crop images of an existing workspace")
    _ = p.add_argument("--name", required=True, help="workspace name")

    p = sub.add_parser(Command.SERVE, help="serve the annotation page locally")
    _ = p.add_argument("--name", required=True, help="workspace name")
    _ = p.add_argument("--open", action="store_true", help="open the page in the default browser")

    p = sub.add_parser(Command.SCORE, help="score a segmentation against the annotations")
    _ = p.add_argument("segmentation", type=Path, help="pybones segmentation output directory")
    _ = p.add_argument("--name", required=True, help="workspace name")
    return parser


def main(argv: "Sequence[str] | None" = None) -> None:
    """Parse and validate the arguments, then run the command (and log it to its own file).

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        invocation = AnnotationInvocation.from_namespace(vars(namespace))
    except ValidationError as exc:
        parser.error(f"invalid arguments:\n{exc}")
    settings, config = load_settings(invocation.settings), load_config(invocation.config)
    _ = configure_logging(config.logging, PROGRAM_NAME)
    args = invocation.args
    workspace = Workspace(Path(settings.workspace_root) / args.name, settings)
    match args:
        case PrepareArgs():
            cmd_prepare(workspace, args.segmentation, args.source, args.force, settings, config)
        case RegenerateArgs():
            cmd_regenerate(workspace, settings, config)
        case ServeArgs():
            cmd_serve(workspace, args.open, settings, config)
        case ScoreArgs():
            cmd_score(workspace, args.segmentation, config)
        case _:
            assert_never(args)


if __name__ == "__main__":
    main()
