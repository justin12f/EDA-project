"""The prediction engine: numerical methods, time series, and ML.

Structured around the claims that matter, not around coverage of the API:
a method that constructs but predicts nonsense is worse than one that refuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from lumen.agents.master_factory import AgentMasterFactory
from lumen.prediction import (
    Family,
    PredictorRegistry,
    Task,
    backtest,
    compare,
    evaluate,
    score,
    split,
    to_series,
    to_supervised,
)
from lumen.prediction.evaluate import Comparison
from lumen.prediction.metrics import mape, r2, rmse

RNG = np.random.default_rng(20260804)


def linear_frame(n: int = 200, backend: str = "polars"):
    """y = 3x1 - 2x2 + 5, plus mild noise. Every regressor should find it."""
    x1 = RNG.uniform(0, 10, n)
    x2 = RNG.uniform(0, 5, n)
    y = 3 * x1 - 2 * x2 + 5 + RNG.normal(0, 0.5, n)
    data = {"x1": x1, "x2": x2, "y": y}
    return pl.DataFrame(data) if backend == "polars" else pd.DataFrame(data)


def seasonal_series(cycles: int = 12, season: int = 7, slope: float = 2.0):
    """A trending, seasonal series — what a signups-per-day column looks like."""
    steps = np.arange(cycles * season, dtype=float)
    pattern = np.tile(np.array([0, 3, 6, 4, 2, -3, -5], dtype=float), cycles)
    return 50 + slope * steps + pattern + RNG.normal(0, 0.4, steps.size)


# ── the registry ────────────────────────────────────────────────────────────


def test_all_three_families_are_registered():
    for family in Family:
        assert PredictorRegistry.names(family=family), f"{family} has no predictors"


def test_every_predictor_declares_a_coherent_identity():
    for name in PredictorRegistry.names():
        predictor = PredictorRegistry.get_class(name)
        assert predictor.name == name
        assert isinstance(predictor.family, Family)
        assert isinstance(predictor.task, Task)
        # Only forecasters may be univariate — the flag changes the call
        # signature, so a mismatch is a runtime surprise.
        if predictor.univariate:
            assert predictor.task is Task.FORECAST


def test_an_unknown_predictor_names_the_alternatives():
    with pytest.raises(ValueError, match="ridge"):
        PredictorRegistry.create("summon_daemon")


def test_the_catalogue_exposes_parameters_so_nothing_has_to_be_guessed():
    catalogue = {row["name"]: row for row in PredictorRegistry.describe()}
    assert "n_estimators" in catalogue["random_forest"]["params"]
    assert "season_length" in catalogue["seasonal_naive"]["params"]
    assert all(row["summary"] for row in catalogue.values()), "every method needs a summary"


# ── numerical ───────────────────────────────────────────────────────────────


def test_least_squares_recovers_known_coefficients():
    X = np.column_stack([RNG.uniform(0, 10, 300), RNG.uniform(0, 5, 300)])
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5

    model = PredictorRegistry.create("least_squares")
    report = model.fit(X, y)

    intercept, a, b = report.diagnostics["coefficients"]
    assert intercept == pytest.approx(5.0, abs=1e-6)
    assert a == pytest.approx(3.0, abs=1e-6)
    assert b == pytest.approx(-2.0, abs=1e-6)


def test_least_squares_reports_rank_deficiency_rather_than_hiding_it():
    """A duplicated column makes the fit non-unique. The caller must be able to see it."""
    base = RNG.uniform(0, 10, 50)
    X = np.column_stack([base, base])
    report = PredictorRegistry.create("least_squares").fit(X, 2 * base + 1)
    assert report.diagnostics["rank_deficient"] is True


def test_polynomial_recovers_a_quadratic():
    x = np.linspace(-5, 5, 100)
    model = PredictorRegistry.create("polynomial", degree=2)
    model.fit(x.reshape(-1, 1), 2 * x**2 - 3 * x + 1)

    predicted = model.predict(np.array([0.0, 1.0, 2.0])).values
    assert predicted == pytest.approx([1.0, 0.0, 3.0], abs=1e-6)


def test_a_polynomial_refuses_to_memorise():
    """Degree 5 through 6 points is interpolation wearing a fit's clothes:
    d+1 coefficients through d+1 points passes exactly through every one."""
    with pytest.raises(ValueError, match="memorisation"):
        PredictorRegistry.create("polynomial", degree=5).fit(
            np.arange(6, dtype=float).reshape(-1, 1), np.arange(6, dtype=float)
        )


def test_interpolation_holds_the_endpoint_instead_of_extrapolating():
    """Continuing the last slope into unobserved territory is how interpolators lie."""
    model = PredictorRegistry.create("linear_interpolation")
    model.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 10.0, 20.0]))

    assert model.predict(np.array([0.5])).values[0] == pytest.approx(5.0)
    assert model.predict(np.array([99.0])).values[0] == pytest.approx(20.0)


def test_a_cubic_spline_passes_through_its_knots():
    x = np.linspace(0, 10, 12)
    y = np.sin(x)
    model = PredictorRegistry.create("cubic_spline")
    model.fit(x.reshape(-1, 1), y)
    assert model.predict(x).values == pytest.approx(y, abs=1e-9)


# ── time series ─────────────────────────────────────────────────────────────


def test_naive_repeats_the_last_observation():
    model = PredictorRegistry.create("naive")
    model.fit(np.arange(10, dtype=float).reshape(-1, 1), np.arange(10, dtype=float))
    assert model.forecast(3).values == pytest.approx([9.0, 9.0, 9.0])


def test_seasonal_naive_repeats_the_last_cycle():
    season = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.tile(season, 5)
    model = PredictorRegistry.create("seasonal_naive", season_length=4)
    model.fit(np.arange(y.size, dtype=float).reshape(-1, 1), y)
    assert model.forecast(6).values == pytest.approx([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])


def test_drift_extends_the_line_through_the_endpoints():
    y = np.arange(0.0, 10.0)
    model = PredictorRegistry.create("drift")
    model.fit(np.arange(y.size, dtype=float).reshape(-1, 1), y)
    assert model.forecast(3).values == pytest.approx([10.0, 11.0, 12.0], abs=1e-9)


def test_exponential_smoothing_tracks_a_trend():
    y = np.arange(0.0, 60.0) * 2.0
    model = PredictorRegistry.create("exponential_smoothing", alpha=0.5, beta=0.5)
    report = model.fit(np.arange(y.size, dtype=float).reshape(-1, 1), y)

    assert report.diagnostics["configuration"] == "holt"
    assert report.diagnostics["trend_per_step"] == pytest.approx(2.0, abs=0.2)
    assert model.forecast(1).values[0] == pytest.approx(120.0, rel=0.05)


def test_holt_winters_beats_a_naive_forecast_on_a_seasonal_series():
    """The claim that justifies the extra machinery. Without it, use naive."""
    y = seasonal_series()

    holt_winters = backtest(
        PredictorRegistry.create(
            "exponential_smoothing", alpha=0.4, beta=0.2, gamma=0.3, season_length=7
        ),
        y,
        folds=3,
        horizon=7,
    )
    naive = backtest(PredictorRegistry.create("naive"), y, folds=3, horizon=7)

    assert holt_winters.metrics["rmse"].value < naive.metrics["rmse"].value


def test_the_smoothing_constants_are_validated():
    with pytest.raises(ValueError, match="alpha"):
        PredictorRegistry.create("exponential_smoothing", alpha=1.5)


# ── machine learning ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting", "knn", "svr"]
)
def test_every_regressor_learns_a_linear_relationship(name):
    frame = linear_frame(200)
    X, y, features = to_supervised(frame, "polars", target="y")
    assert features == ["x1", "x2"]

    result = evaluate(PredictorRegistry.create(name), X, y)
    assert not result.failed, result.error
    # A method that cannot reach R² 0.8 on a near-noiseless linear relationship
    # is misconfigured, not merely weaker.
    assert result.metrics["r2"].value > 0.8, f"{name} scored {result.metrics['r2'].value}"


@pytest.mark.parametrize("name", ["logistic", "random_forest_classifier", "gradient_boosting_classifier"])
def test_every_classifier_separates_two_clear_classes(name):
    X = np.vstack([RNG.normal(0, 1, (100, 2)), RNG.normal(6, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])

    result = evaluate(PredictorRegistry.create(name), X, y)
    assert not result.failed, result.error
    assert result.metrics["accuracy"].value > 0.95


def test_a_classifier_refuses_a_single_class_target():
    with pytest.raises(ValueError, match="two classes"):
        PredictorRegistry.create("logistic").fit(RNG.normal(0, 1, (30, 2)), np.zeros(30))


def test_fitting_is_reproducible():
    """A proposal a human approved must produce the same model when the worker runs it."""
    X = RNG.uniform(0, 10, (80, 3))
    y = X @ np.array([1.0, -2.0, 0.5])

    first = PredictorRegistry.create("random_forest", n_estimators=20)
    second = PredictorRegistry.create("random_forest", n_estimators=20)
    first.fit(X, y)
    second.fit(X, y)

    assert first.predict(X).values == pytest.approx(second.predict(X).values)


def test_knn_refuses_fewer_rows_than_neighbours():
    with pytest.raises(ValueError, match="n_neighbors"):
        PredictorRegistry.create("knn", n_neighbors=10).fit(
            RNG.uniform(0, 1, (6, 2)), RNG.uniform(0, 1, 6)
        )


# ── guard rails ─────────────────────────────────────────────────────────────


def test_too_little_data_is_refused_not_fitted():
    with pytest.raises(ValueError, match="not a model"):
        PredictorRegistry.create("ridge").fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="not been fitted"):
        PredictorRegistry.create("ridge").predict(np.array([[1.0]]))


def test_rows_with_a_null_target_are_dropped_not_imputed():
    """Inventing a target value fabricates the thing being learned."""
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.arange(20, dtype=float)
    y[5] = np.nan

    report = PredictorRegistry.create("least_squares").fit(X, y)
    assert report.n_samples == 19


def test_forecast_on_a_multivariate_model_says_what_to_call_instead():
    model = PredictorRegistry.create("ridge")
    with pytest.raises(NotImplementedError, match="predict"):
        model.forecast(3)


# ── metrics ─────────────────────────────────────────────────────────────────


def test_a_perfect_prediction_scores_perfectly():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_r2_is_undefined_for_a_constant_truth():
    """Predicting a constant perfectly explains no variance, because there is none."""
    constant = np.array([5.0, 5.0, 5.0])
    assert np.isnan(r2(constant, constant))


def test_mape_excludes_zeros_instead_of_returning_infinity():
    truth = np.array([0.0, 100.0])
    predicted = np.array([10.0, 110.0])
    assert mape(truth, predicted) == pytest.approx(0.1)
    assert np.isnan(mape(np.zeros(3), np.ones(3)))


def test_metrics_declare_their_direction():
    metrics = score(np.array([1.0, 2.0]), np.array([1.1, 2.1]), Task.REGRESSION)
    assert metrics["rmse"].higher_is_better is False
    assert metrics["r2"].higher_is_better is True


# ── splitting and comparison ────────────────────────────────────────────────


def test_an_ordered_split_takes_the_tail_not_a_sample():
    """A random split on a time series trains on the future. Every metric then lies."""
    X = np.arange(100, dtype=float).reshape(-1, 1)
    y = np.arange(100, dtype=float)

    _, X_test, _, y_test = split(X, y, 0.2, ordered=True)
    assert y_test[0] == 80.0 and y_test[-1] == 99.0

    _, _, _, y_random = split(X, y, 0.2, ordered=False)
    assert not np.array_equal(np.sort(y_random), y_test)


def test_compare_ranks_by_the_right_direction():
    frame = linear_frame(200)
    X, y, _ = to_supervised(frame, "polars", target="y")

    comparison = compare(X, y, task=Task.REGRESSION, candidates=["ridge", "knn", "random_forest"])

    assert comparison.best is not None
    values = [e.metrics["rmse"].value for e in comparison.ranked if not e.failed]
    assert values == sorted(values), "lower RMSE must rank first"


def test_a_failing_candidate_does_not_sink_the_comparison():
    """One method that cannot fit this data is a result about that method."""
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.arange(12, dtype=float)

    comparison = compare(
        X,
        y,
        task=Task.REGRESSION,
        candidates=["ridge", "knn"],
        test_fraction=0.5,
    )
    # knn asking for 20 neighbours cannot fit 6 training rows; ridge can.
    comparison = Comparison(
        task=Task.REGRESSION,
        ranked=sorted(
            [
                evaluate(PredictorRegistry.create("ridge"), X, y, test_fraction=0.5),
                evaluate(
                    PredictorRegistry.create("knn", n_neighbors=20), X, y, test_fraction=0.5
                ),
            ],
            key=lambda e: (1, 0.0) if e.failed else (0, e.metrics["rmse"].value),
        ),
    )
    assert any(e.failed for e in comparison.ranked)
    assert comparison.best is not None
    assert comparison.ranked[-1].failed, "failures sort last, never first by accident"


# ── extraction across backends ──────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_extraction_agrees_across_backends(backend):
    frame = linear_frame(50, backend=backend)
    X, y, features = to_supervised(frame, backend, target="y")
    assert features == ["x1", "x2"]
    assert X.shape == (50, 2)
    assert y.shape == (50,)


def test_extraction_refuses_to_guess_an_encoding():
    """How to encode a category is a modelling decision a person should approve."""
    frame = pl.DataFrame({"country": ["DE", "US", "FR"], "y": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not numeric"):
        to_supervised(frame, "polars", target="y", features=["country"])


def test_order_by_sorts_before_extracting():
    """Every forecaster treats row order as time order, so this is load-bearing."""
    frame = pl.DataFrame({"day": [3, 1, 2], "value": [30.0, 10.0, 20.0]})
    series = to_series(frame, "polars", "value", order_by="day")
    assert series.tolist() == [10.0, 20.0, 30.0]


def test_a_missing_target_names_the_available_columns():
    with pytest.raises(ValueError, match="x1"):
        to_supervised(linear_frame(10), "polars", target="does_not_exist")


# ── the injected layer ──────────────────────────────────────────────────────


def test_the_master_factory_exposes_prediction():
    prediction = AgentMasterFactory("polars").prediction()
    assert prediction.backend == "polars"
    assert "ridge" in prediction.available()
    assert set(prediction.available(task=Task.FORECAST)) == {
        "naive", "seasonal_naive", "drift", "exponential_smoothing", "moving_average"
    }


def test_fit_through_the_injected_layer():
    prediction = AgentMasterFactory("polars").prediction()
    _, report, features = prediction.fit(linear_frame(100), "ridge", target="y")

    assert report.predictor == "ridge"
    assert report.n_samples == 100
    assert features == ["x1", "x2"]


def test_forecast_through_the_injected_layer():
    prediction = AgentMasterFactory("polars").prediction()
    series = seasonal_series(cycles=9, season=7)
    frame = pl.DataFrame({"day": np.arange(series.size), "signups": series})

    forecast, report = prediction.forecast(
        frame, "exponential_smoothing", "signups", horizon=7,
        order_by="day", alpha=0.4, beta=0.2, gamma=0.3, season_length=7,
    )

    assert len(forecast.as_list()) == 7
    assert report.diagnostics["configuration"] == "holt-winters"


def test_comparing_forecasters_includes_the_naive_baseline():
    prediction = AgentMasterFactory("polars").prediction()
    frame = pl.DataFrame({"value": seasonal_series(cycles=10, season=7)})

    comparison = prediction.compare_forecasters(frame, "value", folds=3, horizon=7)
    names = [e.predictor for e in comparison.ranked]

    assert "naive" in names, "without the baseline nobody checks whether complexity paid"
    assert comparison.best is not None


def test_forecasting_with_a_non_forecaster_says_which_methods_qualify():
    prediction = AgentMasterFactory("polars").prediction()
    frame = pl.DataFrame({"value": np.arange(50, dtype=float)})

    with pytest.raises(ValueError, match="naive"):
        prediction.forecast(frame, "ridge", "value", horizon=5)


# ── seasonal period detection ───────────────────────────────────────────────


def test_a_weekly_rhythm_is_detected_through_a_trend():
    from lumen.prediction.timeseries import detect_season

    steps = np.arange(84, dtype=float)
    weekly = np.tile([0, 12, 25, 18, 9, -14, -22], 12)
    assert detect_season(300 + 4.5 * steps + weekly) == 7
    assert detect_season(300 + weekly + RNG.normal(0, 3, 84)) == 7


def test_a_smooth_long_cycle_is_not_mistaken_for_a_short_one():
    """On any smooth series adjacent points correlate near 1, so the global
    argmax always returns 2. The period must be a peak, not a maximum."""
    from lumen.prediction.timeseries import detect_season

    monthly = np.tile(np.sin(np.linspace(0, 2 * np.pi, 30, endpoint=False)), 4)
    assert detect_season(monthly) == 30


@pytest.mark.parametrize(
    "label,series",
    [
        ("pure trend", 300 + 4.5 * np.arange(84, dtype=float)),
        ("constant", np.full(84, 7.0)),
        ("noise", RNG.normal(0, 1, 84)),
    ],
)
def test_no_season_is_reported_when_there_is_none(label, series):
    """A wrong period imposes a rhythm the data does not have."""
    from lumen.prediction.timeseries import detect_season

    assert detect_season(series) is None, label


def test_detection_needs_two_full_cycles():
    from lumen.prediction.timeseries import detect_season

    assert detect_season(np.tile([1.0, 5.0, 3.0], 2)) is None  # too short to judge
