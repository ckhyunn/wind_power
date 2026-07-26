import numpy as np

from src.calibration import apply_calibrator, fit_calibrator


def test_calibrator_corrects_systematic_multiplicative_bias():
    rng = np.random.default_rng(0)
    actual = rng.uniform(0, 1, size=500)
    pred = actual * 1.2  # model systematically over-predicts by 20%

    calibrator = fit_calibrator(pred, actual)
    calibrated = apply_calibrator(calibrator, pred)

    raw_error = np.mean(np.abs(pred - actual))
    calibrated_error = np.mean(np.abs(calibrated - actual))
    assert calibrated_error < raw_error


def test_calibrator_is_monotonic_in_raw_prediction():
    rng = np.random.default_rng(1)
    actual = rng.uniform(0, 1, size=300)
    pred = actual + rng.normal(0, 0.05, size=300)

    calibrator = fit_calibrator(pred, actual)
    grid = np.linspace(0, 1, 50)
    calibrated_grid = apply_calibrator(calibrator, grid)
    assert np.all(np.diff(calibrated_grid) >= 0)


def test_calibrator_clips_out_of_range_predictions_instead_of_extrapolating():
    actual = np.array([0.1, 0.2, 0.3, 0.4])
    pred = np.array([0.1, 0.2, 0.3, 0.4])

    calibrator = fit_calibrator(pred, actual)
    # far outside the training range seen by the calibrator
    calibrated = apply_calibrator(calibrator, np.array([-5.0, 50.0]))
    assert calibrated[0] == calibrator.predict([0.1])[0]
    assert calibrated[1] == calibrator.predict([0.4])[0]
