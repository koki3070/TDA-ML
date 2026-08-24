"""Named numerical floors (document in paper Methods / supplement).

These are part of the declared numerical method, not silent patches for
degenerate intermediates. Prefer hard-fail over inventing new undeclared floors.

Declared FP guard without a constant: squared distances are clamped at 0 before
masked ``sqrt`` (``tda_ml.topology._sqrt_off_diagonal_only``) to strip negative
floating-point rounding noise from a mathematically non-negative quantity.
"""

# Denominators in metric tensor, probability weighting, aspect ratio.
NUMERICAL_EPS = 1e-8

# Ridge on local 2x2 covariance before ``eigh``.
PCA_RIDGE_EPS = 1e-6

# Floor before ``sqrt`` on PCA eigenvalues (non-negative spectrum).
EIGENVALUE_FLOOR = 1e-6

# Lower clamp on inlier probability in Mahalanobis distance weighting.
INLIER_PROB_MIN = 1e-4

# Absolute-sum threshold to drop zero-padded rows in fixed-size point tensors.
# Dataset loaders pad short clouds with zeros; this is the declared mask, not a
# modeling floor on physical coordinates.
ZERO_PAD_ABS_SUM = 1e-6

# Relative axis gap below which an ellipse is treated as circular (no unique
# major-axis direction). Recorded in the run manifest as a declared constant.
ORIENTATION_MIN_AXIS_GAP = 1e-3

# Minimum pairwise center separation for ellphi-compatible clouds.
# Two nearly coincident cloud points make the ellphi tangency derivative w.r.t.
# the ellipse center (mu) numerically undefined; samplers reject candidates
# closer than this rather than emitting degenerate clouds.
MIN_ELLPHI_CENTER_SEPARATION = 1e-2
# Backward-compatible alias (tangent outlier sampler historically used this name).
MIN_TANGENT_OUTLIER_SEPARATION = MIN_ELLPHI_CENTER_SEPARATION
