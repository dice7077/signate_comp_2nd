from collections import OrderedDict

from .assign_data_id import assign_data_id
from .split_signate_by_type import split_signate_by_type

STEP_REGISTRY = OrderedDict(
    [
        ("assign_data_id", assign_data_id),
        ("split_signate_by_type", split_signate_by_type),
    ]
)

__all__ = [
    "assign_data_id",
    "split_signate_by_type",
    "STEP_REGISTRY",
]
