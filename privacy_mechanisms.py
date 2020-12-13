# coding: utf-8

"""Code for the Privacy Mechanisms described in the following paper:

G. Theodorakopoulos, E. Panaousis, K. Liang and G. Loukas, 
'On-the-fly Privacy for Location Histograms' 
in IEEE Transactions on Dependable and Secure Computing, 2020.

no_protection_privacy (Section 5.1).
no_knowledge_protection_privacy (Section 5.2).
heatmap_knowledge_privacy (Section 5.3 for c>0; Section 5.4 for c=0)."""

from scipy.optimize import minimize
from scipy.stats import power_divergence
import numpy as np


def dp(h1, h2):
    """Privacy distance between histograms `h1` and `h2`
    `h1` and `h2` contain counts, not probabilities."""

    # [0] returns statistic
    # [1] returns p-value
    return power_divergence(h1, f_exp=h2, lambda_="pearson")[0]


def phist(trajectory, N_LOCS: int):
    """Yields a sequence of histograms, each of length `N_LOCS`, 
    formed by larger and larger prefixes of locations from `trajectory`.
    
    Each histogram in the sequence is an np.array with non-negative integers."""

    partial_hist = np.zeros((N_LOCS, 1), dtype=np.int)
    for loc in trajectory:
        partial_hist[loc] += 1
        yield partial_hist


def no_protection_privacy(trajectory, target_pdf, N_LOCS: int):
    """Privacy cost and quality loss for a Privacy Mechanism that
    transmits the user's true location unmodified at each time instant.
    Described in Section 5.1 of the paper.

    `trajectory` (np.array):    sequence of locations that the user visits
    `target_pdf` (np.array):    target probability distribution function. 
                                The privacy cost dp is computed between this (converted to a histogram) 
                                and the partial histogram at each time"""

    privacy = [
        dp(ph, time * target_pdf)[0] for time, ph in enumerate(phist(trajectory, N_LOCS), start=1)
    ]

    # Calculate privacy as the worst case dp distance across all times.
    # In both cases, the quality loss is 0.0, because we transmit the user's true location at each time instant.
    # For RESEMBLE, the worst case is the largest distance from the target that we want to resemble:
    return max(privacy), 0.0
    # For DIFFER, the worst case is the shortest distance from the target that we want to differ from:
    # return min(privacy), 0.0


def no_knowledge_protection_privacy(trajectory, target_pdf, N_LOCS: int, dq, c):
    """Privacy cost and quality loss for a Privacy Mechanism that 
    has no knowledge of the user's mobility
    and enforces the privacy constraint `dp() <= c` (RESEMBLE) or `dp() >= c` (DIFFER) at all times.
    Described in Section 5.2 of the paper.

    `trajectory` (np.array):    sequence of locations that the user visits
    `target_pdf` (np.array):    target probability distribution function. 
                                The privacy cost dp is computed between this (converted to a histogram) 
                                and the submitted histogram at each time
    `dq` (np.array)             quality loss matrix when submitting a fake location instead of the true location
    `c` (float)                 privacy parameter. The PM aims to keep dp <= c (for RESEMBLE) or dp >= c (for DIFFER)"""

    hsub = np.zeros((N_LOCS, 1), dtype=np.int)
    quality_loss = 0.0

    for time, loc in enumerate(trajectory, start=1):
        # Examine plocations in ascending order of quality loss
        dq_loc_asc = np.argsort(dq[loc])
        privacy_cost = np.zeros((N_LOCS, 1), dtype=np.float)
        for ploc in dq_loc_asc:
            # Examine if ploc satisfies the privacy constraint
            # If so, update the quality loss and move on to next time instant
            # If not, examine the ploc with the next higher quality loss
            hsub[ploc] += 1
            privacy_cost[ploc] = dp(hsub, time * target_pdf)
            if privacy_cost[ploc] <= c:  # <=c for RESEMBLANCE, >=c for AVOIDANCE
                quality_loss += dq[loc][ploc]
                break
            else:
                hsub[ploc] -= 1
        else:  # No acceptable ploc found. All plocs violate the privacy constraint, so pick the ploc with min violation
            ploc = np.argmin(privacy_cost)
            hsub[ploc] += 1
            quality_loss += dq[loc][ploc]

    return dp(hsub, len(trajectory) * target_pdf)[0], quality_loss


def heatmap_knowledge_privacy(target_pdf, N_LOCS: int, dq, pi, T: int, c, tolerance=1e-3):
    """Privacy cost and quality loss for a Privacy Mechanism that 
    knows the user's mobility as a pdf on locations
    and enforces the privacy constraint `dp() <= c` in expectation at time `T` (the end of the trajectory).
    Described in Section 5.3 of the paper.
    The case where `c=0`  (Perfect Privacy for RESEMBLE) is described in Section 5.4 of the paper.

    `target_pdf` (np.array):    target probability distribution function. 
                                The privacy cost dp is computed between `target_pdf`
                                and the expected frequencies of visited locations at time T
    `dq` (np.array)             quality loss matrix when submitting a fake location instead of the true location
    `pi` (np.array)             user mobility profile. Relative frequencies of the locations visited by the user.
    `T` (int)                   length of the user's trajectory
    `c` (float)                 privacy parameter. The PM aims to keep dp <= c (for RESEMBLE) or dp >= c (for DIFFER)"""

    # Multiply row r of dq with value pi(r)
    rshp = np.reshape
    pi_dq = rshp(pi * dq, (N_LOCS * N_LOCS, 1)).T

    # \sum_{r, r'} \pi(r) f_{rr'} d_q(r,r')
    # `x` is vector formed by concatenating rows of the F matrix, F = f_{rr'}
    q_loss_objective = lambda x: pi_dq @ x

    # \sum_{r'} f_{rr'} <= 1, \forall r
    eq_cons = {
        "type": "ineq",
        "fun": lambda x: np.array([1 - sum(row) for row in x.reshape((N_LOCS, N_LOCS))]),
        "jac": lambda x: -np.kron(np.eye(N_LOCS), np.ones(N_LOCS)),
    }

    # RESEMBLE: sum_{r'} ( sum_r pi(r) f_{rr'} - htarget(r') )^2} / htarget(r') <= c / T
    # DIFFER:   sum_{r'} ( sum_r pi(r) f_{rr'} - htarget(r') )^2} / htarget(r') >= c / T
    ineq_cons = {
        "type": "ineq",  # All inequality constraints are implied to be nonnegative: f(x) >= 0
        "fun": lambda x: c / T
        - np.sum((rshp(x, (N_LOCS, N_LOCS)).T @ pi - target_pdf) ** 2 / target_pdf),
    }

    # Bounds: 0 <= f_{rr'} <= 1
    bnds = tuple((0, 1) for _ in range(N_LOCS * N_LOCS))

    # Random stochastic matrix for initialization
    m = np.random.rand(N_LOCS, N_LOCS)
    x0 = (m / m.sum(axis=1, keepdims=True)).reshape(N_LOCS * N_LOCS, 1) / 2.0

    res = minimize(
        q_loss_objective,
        x0,
        method="SLSQP",
        jac=lambda x: pi_dq,
        constraints=[eq_cons, ineq_cons],
        options={"ftol": tolerance, "disp": True},
        bounds=bnds,
    )

    opt_x = res.x

    # Privacy distance between expected frequencies of visited locations and target_pdf
    privacy = T * np.sum((opt_x.reshape((N_LOCS, N_LOCS)).T @ pi - target_pdf) ** 2 / target_pdf)

    # Quality Loss: expected dq distance between visited locations and true locations
    quality = T * q_loss_objective(opt_x)[0]

    return privacy, quality
