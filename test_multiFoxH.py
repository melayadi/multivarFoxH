import numpy as np
from scipy.special import gamma
from tqdm import tqdm
import matplotlib.pyplot as plt

from multiFoxH import compMultiFoxH

"""
A simple script that illustrates how to use multiFoxH module for computing the multivariate
Fox H function.

In this example, we compute the symbol error rate for the M-QAM modulation with EGC diversity
and EGK fading. For simplicity, we will assume all channels have the same parameters. This 
corresponds to Example 2, page 14 in [1]

[1] Closed-form exact and asymptotic expressions for the symbol error rate and capacity of the 
    $H$-function fading channel, HR Alhennawi, MMH El Ayadi, MH Ismail, HAM Mourad - IEEE 
    Transactions on Vehicular Technology, 2015
"""

if __name__ == '__main__':
    # Channel parameters
    M = 16  # 16-QAM modulation
    L = 4  # No. of diversity receivers
    mu = 1
    xi = 1.5
    mu_s = 3.5
    xi_s = 2

    # Dependent parameters
    beta = gamma(mu + 1 / xi) / gamma(mu)
    beta_s = gamma(mu_s + 1 / xi_s) / gamma(mu_s)

    # Parameters for first term in (46) in [1]
    mn = [(0, 0)] + [(2, 1)] * L
    pq = [(0, 1)] + [(1, 2)] * L
    a = [[(1, 2)]] * L
    b = [[(mu, 1 / xi), (mu_s, 1 / xi_s)]] * L
    c = []
    d = [tuple([0] + [1] * L)]

    # Parameters for second term in (46) in [1]
    mn2 = [(0, 1)] + [(1, 1)] + [(2, 1)] * L
    pq2 = [(1, 1)] + [(1, 2)] + [(1, 2)] * L
    a2 = [[(1, 1)]] + [[(1, 2)]] * L
    b2 = [[(0.5, 1.0), (0, 1.0)]] + [[(mu - 1 / xi, 1 / xi), (mu_s - 1 / xi_s, 1 / xi_s)]] * L
    c2 = [tuple([0.5, 1] + [1] * L)]
    d2 = [tuple([0, 0] + [2] * L)]

    # SNR expression
    k = (2 / gamma(mu_s) / gamma(mu)) ** L * 2 * (np.sqrt(M) - 1) / M / np.pi
    snrs = np.arange(0, 30.1, 5)
    sers = []
    for hg in tqdm(snrs):
        lambda_l = beta * beta_s / (10 ** (hg / 10))
        z1 = [lambda_l * L * (M - 1) / 6] * L
        params1 = z1, mn, pq, c, d, a, b
        H1 = np.real(compMultiFoxH(params1, nsubdivisions=20))

        z2 = [1] + [lambda_l * L * 2 * (M - 1) / 3] * L
        params2 = z2, mn2, pq2, c2, d2, a2, b2
        H2 = np.real(compMultiFoxH(params2, nsubdivisions=20))
        SER = k * (H1 + (np.sqrt(M) - 1) / np.pi * H2)
        sers.append(SER)

    plt.semilogy(snrs, sers)
    plt.show()
