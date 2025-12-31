from collections import OrderedDict

from .assign_data_id import assign_data_id
from .build_tag_id_features import build_tag_id_features
from .drop_sparse_columns import drop_sparse_columns
from .join_koji_price import join_koji_price
from .join_land_price import join_land_price
from .join_population_projection import join_population_projection
from .layout import STEP_LAYOUT
from .split_signate_by_type import split_signate_by_type

_STEP_FUNCTIONS = {
    "assign_data_id": assign_data_id,
    "join_koji_price": join_koji_price,
    "join_land_price": join_land_price,
    "drop_sparse_columns": drop_sparse_columns,
    "join_population_projection": join_population_projection,
    "split_signate_by_type": split_signate_by_type,
    "build_tag_id_features": build_tag_id_features,
}

STEP_REGISTRY = OrderedDict(
    (step_name, _STEP_FUNCTIONS[step_name]) for step_name in STEP_LAYOUT.keys()
)

__all__ = [
    "assign_data_id",
    "build_tag_id_features",
    "join_koji_price",
    "join_land_price",
    "drop_sparse_columns",
    "join_population_projection",
    "split_signate_by_type",
    "STEP_REGISTRY",
    "STEP_LAYOUT",
]
