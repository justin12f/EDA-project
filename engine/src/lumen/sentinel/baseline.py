"""Per-column, self-calibrating baselines — ADR-0013.

Replaces ADR-0008's single global null-rate-delta threshold with one
derived from each column's own history, and adds the three baseline kinds
ADR-0008 never checked at all: a numeric column's value drift, a categorical
column's distribution shift, and a source's row-count drift (freeform-string
and datetime columns are both handled by reducing them to a single derived
number per scan — a string's length, a timestamp column's freshness lag —
and reusing `NumericBaseline` rather than inventing two more mechanisms; see
`services/api/src/lumen_api/baselines.py` for where that reduction happens).

Every baseline here is an *exponentially weighted* rolling statistic, not a
lifetime average. A lifetime mean gets harder to move with every observation
— by the hundredth scan a single new one barely matters, which is the wrong
shape for Action Item 7's two requirements at once: adapt to a genuine,
sustained business change, but still flag a short, sharp incident as
abnormal while it is happening. EWMA does both, tuned by one constant.

Nothing here reads the raw frame or a database — a caller passes in what a
scan already condensed a column down to (a mean, a set of value counts), the
same "the tick stays cheap enough to run against every source" discipline
`drift.py` documents for itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

BaselineKind = Literal["numeric", "categorical", "datetime", "freeform_string"]

# Below this many scans, a column's own history is too short to calibrate
# confidently — a control limit computed from two data points is not
# conservative, it is noise wearing a lab coat. ADR-0013 §3's own words: "a
# cold-start column... falls back to a conservative default until it
# accumulates enough scans to calibrate itself."
MIN_SAMPLE_SIZE = 5

# How many standard deviations outside the mean counts as a breach. Three is
# the standard control-chart convention (Shewhart): under a roughly normal
# distribution this flags well under 1% of in-band scans as false positives,
# which is what "useful from the first scan" requires — a gate that cries
# wolf constantly gets ignored, which is worse than one that occasionally
# misses something small.
DEFAULT_Z = 3.0

# A categorical share moving by more than this many percentage points from
# its own baseline is a shift worth surfacing. A brand-new category is
# reported once it clears this share of the current scan — a single new row
# out of ten thousand is not "a new category appearing at volume" (ADR-0013
# §1's own phrase), it is one row.
DEFAULT_CATEGORY_DEVIATION = 0.15
DEFAULT_NEW_CATEGORY_SHARE = 0.05

_NUMERIC_DTYPE_HINTS = ("int", "float", "double", "decimal", "numeric")
_DATETIME_DTYPE_HINTS = ("date", "time", "timestamp")

# A string column below this fraction of distinct-to-non-null values behaves
# like a fixed vocabulary (a status code, a country) rather than free text —
# the same heuristic a human skimming a `.value_counts()` output would use,
# made deterministic. A five-country column with thousands of rows clears
# this by a wide margin; an email address or a free-text note (near-100%
# unique) does not.
CATEGORICAL_UNIQUE_RATIO = 0.2


def classify_baseline_kind(dtype: str, *, non_null_count: int, distinct_count: int) -> BaselineKind:
    """A column's baseline_kind, from its dtype and this scan's cardinality
    — never from a human choosing one (ADR-0013 §1's entire point)."""
    lowered = dtype.lower()
    if any(hint in lowered for hint in _DATETIME_DTYPE_HINTS):
        return "datetime"
    if any(hint in lowered for hint in _NUMERIC_DTYPE_HINTS):
        return "numeric"
    if non_null_count == 0:
        return "categorical"
    ratio = distinct_count / non_null_count
    return "categorical" if ratio <= CATEGORICAL_UNIQUE_RATIO else "freeform_string"


@dataclass(frozen=True)
class NumericBaseline:
    """A rolling mean/variance of one per-scan summary number — a numeric
    column's mean, a null rate, a string's average length, a datetime
    column's freshness lag, a source's row-count delta. Updated once per
    scan by EWMA, so this never needs the raw history, only the single
    number the previous scan's baseline already condensed it to.
    """

    mean: float = 0.0
    variance: float = 0.0
    sample_size: int = 0

    # 2/ALPHA - 1 is the standard EWMA-to-simple-moving-average equivalence:
    # 0.2 behaves like roughly a 9-scan window. Chosen so a source ticking
    # daily settles on a sustained shift as the new normal within under two
    # weeks, while a single scan still moves the band too little to mask
    # itself as normal (Action Item 7's "still catches a real incident").
    ALPHA = 0.2

    def updated(self, value: float) -> NumericBaseline:
        if self.sample_size == 0:
            return NumericBaseline(mean=value, variance=0.0, sample_size=1)
        deviation = value - self.mean
        mean = self.mean + self.ALPHA * deviation
        # Standard EWMA variance recurrence (Finch 2009): blends the decayed
        # previous variance with this scan's squared deviation from the
        # *new* mean, keeping the update a single pass with no stored history.
        variance = (1 - self.ALPHA) * (self.variance + self.ALPHA * deviation * deviation)
        return NumericBaseline(mean=mean, variance=variance, sample_size=self.sample_size + 1)

    @property
    def stdev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def control_limits(self, z: float = DEFAULT_Z) -> tuple[float, float]:
        band = z * self.stdev
        return (self.mean - band, self.mean + band)

    def is_calibrated(self, *, min_sample_size: int = MIN_SAMPLE_SIZE) -> bool:
        return self.sample_size >= min_sample_size

    def is_within_band(self, value: float, *, z: float = DEFAULT_Z) -> bool:
        """True if `value` is inside the calibrated band, or if there is not
        yet enough history to have a confident opinion — cold start must
        never itself read as an anomaly (ADR-0013's own "Harder" note: "a
        column with three scans of history should not produce a
        confident-looking anomaly alert")."""
        if not self.is_calibrated():
            return True
        low, high = self.control_limits(z)
        return low <= value <= high


@dataclass(frozen=True)
class CategoryShift:
    category: str
    kind: Literal["new_category", "share_shift"]
    current_share: float
    baseline_share: float | None = None


@dataclass(frozen=True)
class CategoricalBaseline:
    """A rolling frequency distribution — each category's EWMA share of the
    column, updated the same way `NumericBaseline` updates a single number,
    just once per observed category."""

    frequencies: dict[str, float] = field(default_factory=dict)
    sample_size: int = 0

    ALPHA = 0.2

    def updated(self, counts: dict[str, int], total: int) -> CategoricalBaseline:
        if total <= 0:
            return self
        current = {category: count / total for category, count in counts.items()}
        if self.sample_size == 0:
            return CategoricalBaseline(frequencies=current, sample_size=1)
        keys = set(self.frequencies) | set(current)
        updated = {
            category: (
                self.frequencies.get(category, 0.0)
                + self.ALPHA * (current.get(category, 0.0) - self.frequencies.get(category, 0.0))
            )
            for category in keys
        }
        return CategoricalBaseline(frequencies=updated, sample_size=self.sample_size + 1)

    def is_calibrated(self, *, min_sample_size: int = MIN_SAMPLE_SIZE) -> bool:
        return self.sample_size >= min_sample_size


def categorical_shift(
    baseline: CategoricalBaseline,
    counts: dict[str, int],
    total: int,
    *,
    deviation_threshold: float = DEFAULT_CATEGORY_DEVIATION,
    new_category_share: float = DEFAULT_NEW_CATEGORY_SHARE,
) -> list[CategoryShift]:
    """"a category appearing far above or below its historical share, or a
    genuinely new category appearing at volume, is the signal" (ADR-0013
    §1) — cold start (below `MIN_SAMPLE_SIZE` scans) reports nothing, same
    reasoning as `NumericBaseline.is_within_band`.
    """
    if total <= 0 or not baseline.is_calibrated():
        return []
    findings: list[CategoryShift] = []
    # Every category either scan knows about, not just this scan's — a
    # category the baseline has seen before that is simply absent this
    # scan is a 100%-to-0% share move, exactly "far below its historical
    # share" (ADR-0013 §1), and would be silently missed by iterating
    # `counts` alone.
    for category in sorted(set(counts) | set(baseline.frequencies)):
        share = counts.get(category, 0) / total
        baseline_share = baseline.frequencies.get(category)
        if baseline_share is None:
            if share >= new_category_share:
                findings.append(CategoryShift(category=category, kind="new_category", current_share=share))
        elif abs(share - baseline_share) >= deviation_threshold:
            findings.append(
                CategoryShift(
                    category=category, kind="share_shift", current_share=share, baseline_share=baseline_share
                )
            )
    return findings
