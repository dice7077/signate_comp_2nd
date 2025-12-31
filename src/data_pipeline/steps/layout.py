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
        ("build_tag_id_features", StepLayout(stage=2, order=1, slug="build_tag_id_features")),
        ("drop_sparse_columns", StepLayout(stage=1, order=1, slug="drop_sparse_columns")),
        (
            "join_population_projection",
            StepLayout(stage=1, order=2, slug="join_population_projection"),
        ),
        ("split_signate_by_type", StepLayout(stage=1, order=3, slug="split_signate_by_type")),
    ]
)


def step_output_dir(step_name: str) -> str:
    try:
        return STEP_LAYOUT[step_name].output_dir_name
    except KeyError as exc:
        raise KeyError(f"Unknown step '{step_name}'") from exc


__all__ = ["STEP_LAYOUT", "StepLayout", "step_output_dir"]

