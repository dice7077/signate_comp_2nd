from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class StepLayout:
    """
    stage: 大分類（フェーズ/ブランチ）
    order: 同一フェーズ内の手続き順
    slug: 既存のステップ識別子
    """

    stage: int
    order: int
    slug: str

    @property
    def prefix(self) -> str:
        return f"{self.stage:02d}_{self.order:02d}"

    @property
    def output_dir_name(self) -> str:
        return f"{self.prefix}_{self.slug}"


STEP_LAYOUT: "OrderedDict[str, StepLayout]" = OrderedDict(
    [
        ("assign_data_id", StepLayout(stage=0, order=1, slug="assign_data_id")),
        ("drop_sparse_columns", StepLayout(stage=1, order=1, slug="drop_sparse_columns")),
        ("split_signate_by_type", StepLayout(stage=1, order=2, slug="split_by_type")),
        (
            "adjust_mansion_unit_area",
            StepLayout(stage=1, order=3, slug="adjust_mansion_unit_area"),
        ),
        (
            "adjust_mansion_money_room",
            StepLayout(stage=1, order=4, slug="adjust_mansion_money_room"),
        ),
        ("build_tag_id_features", StepLayout(stage=2, order=1, slug="build_tag_id_features")),
        ("join_koji_price", StepLayout(stage=3, order=1, slug="join_koji_price")),
        (
            "join_population_projection",
            StepLayout(stage=4, order=1, slug="join_population_projection"),
        ),
        (
            "join_land_price",
            StepLayout(stage=5, order=1, slug="join_land_price"),
        ),
    ]
)


def step_output_dir(step_name: str) -> str:
    try:
        return STEP_LAYOUT[step_name].output_dir_name
    except KeyError as exc:
        raise KeyError(f"Unknown step '{step_name}'") from exc


__all__ = ["STEP_LAYOUT", "StepLayout", "step_output_dir"]

