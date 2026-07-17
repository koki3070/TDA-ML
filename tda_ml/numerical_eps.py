"""Named numerical floors (document in paper Methods / supplement)."""

# Denominators in metric tensor, probability weighting, aspect ratio.
NUMERICAL_EPS = 1e-8

# Ridge on local 2x2 covariance before ``eigh``.
PCA_RIDGE_EPS = 1e-6

# Floor before ``sqrt`` on PCA eigenvalues (non-negative spectrum).
EIGENVALUE_FLOOR = 1e-6

# Lower clamp on inlier probability in Mahalanobis distance weighting.
INLIER_PROB_MIN = 1e-4

# Minimum center separation enforced when sampling tangent-direction outliers.
# Two nearly coincident cloud points make the ellphi tangency derivative w.r.t.
# the ellipse center (mu) numerically undefined; the sampler rejects candidates
# closer than this to any existing point rather than emitting degenerate clouds.
MIN_TANGENT_OUTLIER_SEPARATION = 1e-2
