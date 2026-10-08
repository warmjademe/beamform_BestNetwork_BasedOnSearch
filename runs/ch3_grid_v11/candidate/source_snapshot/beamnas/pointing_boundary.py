"""Endpoint-aware extension; preserve the frozen Chapter 4 interior-only module."""
import numpy as np

from .pointing import nearest_peak_directions as interior_directions


def nearest_peak_directions(weights, desired, step_deg=.1, batch_size=256):
    weights, desired = np.asarray(weights), np.asarray(desired)
    result = interior_directions(weights, desired, step_deg, batch_size)
    result['legacy_interior_peak_deg'] = result['peak_deg'].copy()
    # Use the scan spacing: a .001-degree endfire neighbor can be identical
    # in float64 power because sin(theta) is stationary at +/-90 degrees.
    grid = np.array([-90., -90.+step_deg, 90.-step_deg, 90.])
    a = np.exp(1j*np.pi*np.sin(np.deg2rad(grid))[:, None]*np.arange(12))
    power = abs(weights.conj() @ a.T)**2
    for angle, eligible in [(-90., power[:, 0] > power[:, 1]), (90., power[:, 3] > power[:, 2])]:
        replace = eligible & (abs(angle-desired) < abs(result['peak_deg']-desired))
        result['peak_deg'][replace] = angle
        result['no_local_peak_fallback'][replace] = False
    result['main_error_deg'] = abs(result['peak_deg']-desired)
    result['endpoint_selected'] = abs(result['peak_deg']) == 90
    return result
