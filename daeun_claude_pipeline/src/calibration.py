from sklearn.isotonic import IsotonicRegression


def fit_calibrator(pred_frac, actual_frac) -> IsotonicRegression:
    """Fit raw model prediction (capacity fraction) -> actual (capacity fraction).

    Isotonic (monotonic, non-parametric) rather than linear: LightGBM's raw output can
    be systematically biased in a non-linear way across the range (e.g. under-predicting
    high-output hours, over-predicting low-output ones) — see docs/submission_log.md for
    the CV evidence this improves both NMAE and FICR. `out_of_bounds="clip"` avoids
    extrapolating past the range of predictions seen during fitting.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pred_frac, actual_frac)
    return calibrator


def apply_calibrator(calibrator: IsotonicRegression, pred_frac):
    return calibrator.predict(pred_frac)
