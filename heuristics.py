# coding: utf-8

"""Produce evaluation results for the following paper:

G. Theodorakopoulos, E. Panaousis, K. Liang and G. Loukas, 
'On-the-fly Privacy for Location Histograms' 
in IEEE Transactions on Dependable and Secure Computing, 2020.

This code loads a mobility dataset and runs the three privacy mechanisms (PMs) described in the paper.
The PMs are imported from privacy_mechanisms.py"""

from collections import Counter
from itertools import groupby
from typing import Any, Dict, Iterable
import numpy as np
import privacy_mechanisms as pms
from distances import quality_distances

### Read in all visit events

# Format of dataset.txt (tab separated columns):
# Column 1: ID of visiting user (int), sorted in ascending order
# Column 2: Semantic label of visited location (str), assumed of length <=25 chars
# Column 3: Latitude of visited location (float)
# Column 4: Longitude of visited location (float)
# Example:
# 210	Clothing Store	40.7242931305695	-73.9976171133222
# 210	Nail Salon	40.727425	-73.990384
# 210	High School	40.928049851945	-74.0605030010955
# 210	Fast Food Restaurant	40.963676	-74.07755655
# 210	Mall	40.9173354418021	-74.0756607055664
# 210	Hardware Store	40.9125072048726	-74.051456451416
# 210	Seafood Restaurant	40.916151417797	-74.0750223219702

# If it is the first time loading the dataset, load directly from the .txt
# The code below loads it twice, once into `events` and once into individual variables (one per attribute)
#
events = np.loadtxt(
    "user210events.txt",
    dtype={"names": ("uid", "loc_label", "lat", "lon"), "formats": ("u4", "U25", "f8", "f8")},
    delimiter="\t",
)
uids, loc_labels, lats, longs = np.loadtxt(
    "user210events.txt",
    dtype={"names": ("uid", "loc_label", "lat", "lon"), "formats": ("u4", "U25", "f8", "f8")},
    delimiter="\t",
    unpack=True,
)

# To save time, save the loaded numpy arrays to `events.npz` and load them directly from that file next time.
# np.savez('events.npz', events=events, uids=uids, loc_labels=loc_labels, lats=lats, longs=longs)

# npzfile = np.load("events.npz")
# events = npzfile["events"]
# uids = npzfile["uids"]  # user IDs, integers starting from 1
# loc_labels = npzfile["loc_labels"]  # location labels, strings (e.g. 'Bar')
# lats = npzfile["lats"]  # latitudes of visited locations, floats
# longs = npzfile["longs"]  # longitudes of visited locations, floats

unique_uids: Iterable[int] = np.unique(uids)  # np.array[int]

### Global target

# Keep the TOP_N most common locations of each user in their trajectory
# Form a target pdf for only these TOP_N locations
TOP_N: int = 4

# Form global target as pdf on a subset of semantic locations
# For this example with TOP_N=4 and target_list as below, we are effectively saying,
# 'Make the top 4 locations of the user approximate equal visits to a Park, Stadium, Bar, Mall'
target_list = "Park Stadium Bar Mall".split()
global_target_pdf = np.ones(len(target_list)) / len(target_list)  # np.array[float]


### Calculate mobility profile and dq matrix for each user from the dataset

## A user's mobility profile (\pi) is the list of relative frequencies of the user's visited locations
## profiles_by_uid[UID] is the profile of user UID

## For each user, the quality loss matrix, dq, is cdist between user's TOP_N locations and the geo-nearest representative
## of each semantic location in a target_list like ['Park', 'Stadium', 'Bar', 'Mall']
## Computed for each user in function quality_distances(from_geo_locs, to_sem_targets)
## dq_by_uid[UID] is the dq matrix of user UID

## Group geolocations by semantic label,
# e.g. geo_locs_by_sem_label['Bar'] = [the (lat, lon) pairs of each Bar in the dataset]
unique_locs_sorted = sorted(set(zip(loc_labels, lats, longs)))
geo_locs_by_sem_label: Dict[str, Any] = {}
for k, v in groupby(unique_locs_sorted, key=lambda elem: elem[0]):
    geo_locs_by_sem_label[k] = [(loc[1], loc[2]) for loc in v]

## Group visit events into trajectories by userid
# For an event `e`: e[1] is the semantic label ('Bar'), and e[2], e[3] are latitude, longitude

sem_trajectories_by_uid: Dict[int, Any] = {
    uid: np.fromiter((e[1] for e in trajectory), dtype="U25")
    for uid, trajectory in groupby(events, lambda e: e["uid"])
}

geo_trajectories_by_uid: Dict[int, Any] = {}
for uid, trajectory in groupby(events, lambda e: e["uid"]):
    geo_trajectories_by_uid[uid] = [(e[2], e[3]) for e in trajectory]


profiles_by_uid: Dict[int, Any] = {}
dq_by_uid: Dict[int, Any] = {}

## Auxiliary dictionaries
# sem_trunc_trajectories_by_uid = {}
geo_trunc_trajectories_by_uid: Dict[int, Any] = {}
enc_trunc_trajectories_by_uid: Dict[int, Any] = {}
n_locs_by_uid: Dict[int, int] = {}
topn_geolocations_by_uid: Dict[int, Any] = {}


def topn(trajectory, num: int):
    """Top `num` most frequent locations in `trajectory` and their relative frequencies"""
    # locs: Top `num` geolocations in the user's trajectory
    # cnts: Integer frequencies of the TOP_N geolocations
    locs_and_counts = Counter((loc[0], loc[1]) for loc in trajectory).most_common(num)
    locs, cnts = tuple(zip(*locs_and_counts))

    return locs, np.array(cnts) / np.sum(cnts)


for uid in unique_uids:
    # sem_traj = sem_trajectories_by_uid[uid]
    geo_traj = geo_trajectories_by_uid[uid]

    topn_geolocations_by_uid[uid], profiles_by_uid[uid] = topn(geo_traj, TOP_N)

    # Filter the user's trajectory to keep only the TOP_N geolocations
    truncated_traj = [loc for loc in geo_traj if loc in topn_geolocations_by_uid[uid]]
    geo_trunc_trajectories_by_uid[uid] = truncated_traj

    # Compute the dq matrix for this user
    dq_by_uid[uid] = quality_distances(
        topn_geolocations_by_uid[uid], target_list, geo_locs_by_sem_label, dec=1
    )

    # Encode the user's trajectory: turn each geolocation to an integer 0, ..., TOP_N-1
    # This encoding must be consistent with the location encoding for the rows of dq and the rows of pi
    # (e.g. use 0 to encode the most common location, etc)
    enc_traj = [topn_geolocations_by_uid[uid].index(loc) for loc in truncated_traj]
    enc_trunc_trajectories_by_uid[uid] = enc_traj
    n_locs_by_uid[uid] = len(topn_geolocations_by_uid[uid])

# Privacy constraint parameters
cs = np.linspace(5.0, 50.0, num=10)

user_stat_results = ["UserID,N_LOCS,T"]
npp_results = ["UserID,privacy,quality"]
nkp_results = ["UserID,c,privacy,quality"]
hm_results = ["UserID,c,privacy,quality"]

###  Run the PMs for a range of User IDs
# for uid in unique_uids:
for uid in [210]:
    # Get the user's `trajectory`, mobility profile `pi`, `target_pdf`, and quality loss matrix `dq`

    trajectory = enc_trunc_trajectories_by_uid[uid]
    T = len(trajectory)

    pi = np.reshape(profiles_by_uid[uid], (-1, 1))
    N_LOCS = n_locs_by_uid[uid]

    target_pdf = np.reshape(global_target_pdf, (-1, 1))

    dq = dq_by_uid[uid]

    assert np.shape(dq) == (TOP_N, TOP_N)

    # Compute and store results
    user_stat_results.append(f"{uid},{N_LOCS},{T}")

    # No protection privacy
    npp_p, npp_q = pms.no_protection_privacy(trajectory, target_pdf, N_LOCS)
    npp_results.append(f"{uid},{npp_p},{npp_q}")

    for c in cs:
        # No knowledge privacy
        nkp_p, nkp_q = pms.no_knowledge_protection_privacy(trajectory, target_pdf, N_LOCS, dq, c)
        nkp_results.append(f"{uid},{c},{nkp_p},{nkp_q}")

        # Heatmap privacy
        hm_p, hm_q = pms.heatmap_knowledge_privacy(target_pdf, N_LOCS, dq, pi, T, c)
        hm_results.append(f"{uid},{c},{hm_p},{hm_q}")


with open("user_stat_results.txt", "w") as f:
    print("\n".join(user_stat_results), file=f, flush=True)
with open("npp_results.txt", "w") as f:
    print("\n".join(npp_results), file=f, flush=True)
with open("nkp_results.txt", "w") as f:
    print("\n".join(nkp_results), file=f, flush=True)
with open("hm_results.txt", "w") as f:
    print("\n".join(hm_results), file=f, flush=True)
