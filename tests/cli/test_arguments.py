"""Tests of the typed command-line arguments (:mod:`pybones.cli.arguments`) and dispatch."""

from pathlib import Path
from typing import TYPE_CHECKING, get_args

import pytest
from pydantic import ValidationError

from pybones.cli import main as cli_main
from pybones.cli.arguments import CommandArgs, Invocation, MeshArgs, PreviewArgs, SegmentArgs
from pybones.cli.main import build_parser, run_command
from pybones.config.loader import load_config
from pybones.segmentation.labels import Label

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

#: Minimal valid argument lists, one per sub-command.
EXAMPLE_ARGV = {
    "config": ("config",),
    "segment": ("segment", "data.zip", "-o", "out"),
    "preview": ("preview", "seg", "-z", "0", "-o", "p.png"),
    "mesh": ("mesh", "seg", "--label", "vessel", "-o", "v.stl"),
    "export": ("export", "seg", "--label", "vessel", "-o", "out"),
    "compare": ("compare", "seg", "--label", "vessel", "--reference", "ref"),
    "quantify": ("quantify", "seg"),
    "serve": ("serve",),
}


def parse(*argv: str) -> Invocation:
    """Parse and validate a command line.

    Args:
        *argv: Command-line arguments.

    Returns:
        The validated invocation.
    """
    return Invocation.from_namespace(vars(build_parser().parse_args(argv)))


def command_models() -> "list[type[BaseModel]]":
    """List the argument models of the :data:`CommandArgs` union.

    Returns:
        One model per sub-command.
    """
    return list(get_args(get_args(CommandArgs)[0]))


def test_every_command_is_parsed_and_dispatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every argument model is reachable from the command line and runs its own command."""
    called: list[str] = []

    def recorder(name: str) -> "Callable[..., None]":
        """Build a stand-in for a ``cmd_*`` function.

        Args:
            name: Sub-command name.

        Returns:
            A function recording ``name`` when called.
        """

        def record(*_args: object) -> None:
            """Record the call.

            Args:
                *_args: Arguments of the command (ignored).
            """
            called.append(name)

        return record

    for name in EXAMPLE_ARGV:
        monkeypatch.setattr(cli_main, f"cmd_{name}", recorder(name))
    config = load_config()
    parsed = [parse(*argv).args for argv in EXAMPLE_ARGV.values()]
    assert {type(args) for args in parsed} == set(command_models())
    for args in parsed:
        run_command(args, config)
    assert called == list(EXAMPLE_ARGV)


def test_global_options_and_typed_arguments() -> None:
    """Global options and sub-command arguments are validated into typed fields."""
    inv = parse("--set", "pipeline.workers=1", "mesh", "seg", "--label", "vessel", "-o", "v.stl")
    assert inv.overrides == ("pipeline.workers=1",)
    assert inv.config is None
    assert isinstance(inv.args, MeshArgs)
    assert inv.args.label is Label.VESSEL
    assert inv.args.segmentation == Path("seg")


def test_mesh_output_must_be_a_mesh_format() -> None:
    """``pybones mesh`` accepts .stl and .glb outputs and refuses anything else."""
    for name in ("v.stl", "v.glb"):
        args = parse("mesh", "seg", "--label", "vessel", "-o", name).args
        assert isinstance(args, MeshArgs)
    with pytest.raises(ValidationError, match="unsupported mesh format"):
        _ = parse("mesh", "seg", "--label", "vessel", "-o", "v.obj")


def test_optional_arguments_default_to_none() -> None:
    """Options not given on the command line are ``None``."""
    inv = parse(*EXAMPLE_ARGV["preview"])
    assert isinstance(inv.args, PreviewArgs)
    assert (inv.args.source, inv.args.scale) == (None, None)


@pytest.mark.parametrize(
    "argv",
    [
        ("preview", "seg", "-z", "-1", "-o", "p.png"),  # negative slice index
        ("preview", "seg", "-z", "0", "-o", "p.png", "--scale", "0"),  # scale must be > 0
        ("mesh", "seg", "--label", "background", "-o", "x.stl"),  # not a foreground label
        ("export", "seg", "--label", "marrow", "-o", "out"),  # unknown label
    ],
)
def test_invalid_arguments_are_rejected(argv: tuple[str, ...]) -> None:
    """Out-of-range values and invalid labels fail validation."""
    with pytest.raises(ValidationError):
        _ = parse(*argv)


def test_arguments_are_immutable() -> None:
    """Validated arguments cannot be modified."""
    inv = parse(*EXAMPLE_ARGV["segment"])
    assert isinstance(inv.args, SegmentArgs)
    with pytest.raises(ValidationError):
        inv.args.output = Path("elsewhere")
