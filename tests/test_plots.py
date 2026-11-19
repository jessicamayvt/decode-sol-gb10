from dataclasses import fields
from pathlib import Path
from typing import Any

from decode_sol.analyze import PointSummary
from decode_sol.plots import BATCH_SIZES, COLOR_BY_MODEL, save_all_figures


def make_summary(*, model: str, kv_cache_dtype: str, batch_size: int) -> PointSummary:
    """A context-512 point with every number set to something plottable on a log axis."""
    values: dict[str, Any] = {}
    for field in fields(PointSummary):
        parse_value = field.type
        assert callable(parse_value)
        values[field.name] = parse_value(1)
    values.update(
        model=model, kv_cache_dtype=kv_cache_dtype, batch_size=batch_size, context_len=512
    )
    values.update(efficiency=0.8, num_repeats=batch_size % 3 + 1)
    return PointSummary(**values)


def test_every_figure_is_written_as_a_png(tmp_path: Path) -> None:
    summaries: list[PointSummary] = []
    for model in COLOR_BY_MODEL:
        for kv_cache_dtype in ("auto", "fp8"):
            for batch_size in BATCH_SIZES:
                summaries.append(
                    make_summary(model=model, kv_cache_dtype=kv_cache_dtype, batch_size=batch_size)
                )

    saved_paths = save_all_figures(summaries, tmp_path)

    assert len(saved_paths) == 3
    for path in saved_paths:
        assert path.read_bytes().startswith(b"\x89PNG")
