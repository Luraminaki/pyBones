"""The slice stacks of the data folder."""

from fastapi import APIRouter

from pybones.server.context import Context
from pybones.server.schemas import DatasetInfo

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets(context: Context) -> list[DatasetInfo]:
    """List the stacks that can be segmented.

    Args:
        context: Server state.

    Returns:
        One description per stack of the data folder, sorted by id.
    """
    return context.library.datasets()
