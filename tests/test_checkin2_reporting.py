from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_checkin2_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    import make_figures

    make_figures._style()
    make_figures.math500_base_figure(RESULTS, tmp_path / "fig3.png")
    make_figures.kuhn_base_figure(RESULTS, tmp_path / "fig4.png")
    assert (tmp_path / "fig3.png").stat().st_size > 10_000 and (tmp_path / "fig4.png").stat().st_size > 10_000
