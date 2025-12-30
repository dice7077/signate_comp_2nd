from collections import OrderedDict

from .assign_data_id import assign_data_id
from .build_tag_id_features import build_tag_id_features
from .drop_sparse_columns import drop_sparse_columns
from .join_population_projection import join_population_projection
from .split_signate_by_type import split_signate_by_type

STEP_REGISTRY = OrderedDict(
    [
        ("assign_data_id", assign_data_id),
        ("build_tag_id_features", build_tag_id_features),
        ("drop_sparse_columns", drop_sparse_columns),
        ("join_population_projection", join_population_projection),
        ("split_signate_by_type", split_signate_by_type),
    ]
)

__all__ = [
    "assign_data_id",
    "build_tag_id_features",
    "drop_sparse_columns",
    "join_population_projection",
    "split_signate_by_type",
    "STEP_REGISTRY",
]
