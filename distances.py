# coding: utf-8

"""Haversine distance, used to compute quality distances.
Quality distances give the quality loss when submitting a fake location instead of the true one."""

import numpy as np
from scipy.spatial.distance import cdist


def haversine(lonlat1, lonlat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    `lonlat1` and `lonlat2` must be of shape (N, 2).    
    Source: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas"""

    lon1, lat1 = lonlat1
    lon2, lat2 = lonlat2
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def quality_distances(from_geo_locs, to_sem_targets, geo_locs_by_sem_label, dec: int):
    """For each geo-location in `from_geo_locs` and sem-label in `to_sem_targets`,
    compute the haversine distance from the geo-location to its nearest geo-location 
    with that sem-label. Distances are rounded to `dec` decimal digits.
    `from_geo_locs`: list of (float, float)
    `to_sem_targets`: list of str
    `geo_locs_by_sem_label`: dict of str -> list of (float, float)"""

    min_distances = np.empty((len(from_geo_locs), len(to_sem_targets)))
    for idx, trg in enumerate(to_sem_targets):
        all_distances = cdist(from_geo_locs, geo_locs_by_sem_label[trg], metric=haversine)
        np.min(all_distances, axis=1, out=min_distances[:, idx])
    return np.around(min_distances, decimals=dec)
