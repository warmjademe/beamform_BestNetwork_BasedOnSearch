"""Nearest-local-peak pointing rule adopted from Chapter 4 on 2026-10-08."""
import numpy as np


def interior_peak_indices(values):
    """Match scipy.signal.find_peaks without height/prominence constraints.

    For a flat-topped peak, use its middle index rounded down. Endpoints are
    excluded. Nonfinite patterns must be reported, not silently discarded.
    """
    values = np.asarray(values)
    assert values.ndim == 1 and np.isfinite(values).all()
    if len(values) < 3:
        return np.empty(0, dtype=int)
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[starts[1:]-1, len(values)-1]
    interior = (starts > 0) & (ends < len(values)-1)
    starts, ends = starts[interior], ends[interior]
    keep = (values[starts] > values[starts-1]) & (values[ends] > values[ends+1])
    return (starts[keep] + ends[keep]) // 2


def nearest_peak(pattern, grid, target):
    peaks = interior_peak_indices(pattern)
    fallback = not len(peaks)
    index = int(np.argmax(pattern)) if fallback else int(peaks[np.argmin(abs(grid[peaks]-target))])
    return float(grid[index]), fallback


def nearest_peak_directions(weights, desired, step_deg=.1, batch_size=256):
    """Chapter 3 unit-element ULA, Chapter 4 extraction rule, all scenes."""
    weights, desired = np.asarray(weights), np.asarray(desired)
    assert weights.shape == (len(desired), 12)
    grid = np.linspace(-90, 90, round(180/step_deg)+1)
    a = np.exp(1j*np.pi*np.sin(np.deg2rad(grid))[:, None]*np.arange(12))
    found, global_found, fallback = [], [], []
    for start in range(0, len(weights), batch_size):
        power = abs(weights[start:start+batch_size].conj() @ a.T)**2
        db = 10*np.log10(power/(power.max(axis=1, keepdims=True)+1e-12)+1e-12)
        for offset, pattern in enumerate(db):
            angle, missing = nearest_peak(pattern, grid, desired[start+offset])
            found.append(angle)
            global_found.append(grid[pattern.argmax()])
            fallback.append(missing)
    return {'peak_deg':np.array(found),'main_error_deg':abs(np.array(found)-desired),
            'global_peak_deg':np.array(global_found),
            'global_error_deg':abs(np.array(global_found)-desired),
            'no_local_peak_fallback':np.array(fallback, dtype=bool)}
