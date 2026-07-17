"""Named numerical floors (document in paper Methods / supplement)."""

# Denominators in metric tensor, probability weighting, aspect ratio.
NUMERICAL_EPS = 1e-8

# Ridge on local 2x2 covariance before ``eigh``.
PCA_RIDGE_EPS = 1e-6

# Floor before ``sqrt`` on PCA eigenvalues (non-negative spectrum).
EIGENVALUE_FLOOR = 1e-6

# Lower clamp on inlier probability in Mahalanobis distance weighting.
INLIER_PROB_MIN = 1e-4

# Minimum pairwise center separation for ellphi-compatible clouds.
# Two nearly coincident cloud points make the ellphi tangency derivative w.r.t.
# the ellipse center (mu) numerically undefined; samplers reject candidates
# closer than this rather than emitting degenerate clouds.
MIN_ELLPHI_CENTER_SEPARATION = 1e-2
# Backward-compatible alias (tangent outlier sampler historically used this name).
MIN_TANGENT_OUTLIER_SEPARATION = MIN_ELLPHI_CENTER_SEPARATION
