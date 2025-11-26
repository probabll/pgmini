import numpy as np

def _split_chains(arr):
    """
    Split each chain into two halves (discard last point if odd length).
    arr: shape (n_chains, n_samples)
    returns shape (2*n_chains, n_samples/2)
    """
    arr = np.asarray(arr)
    m, n = arr.shape
    half = n // 2
    left  = arr[:, :half]
    right = arr[:, n-half:]
    return np.vstack([left, right])


def rhat_classic(arr):
    """
    Classic (non-split) Gelman–Rubin R-hat.
    arr: shape (n_chains, n_samples)
    """
    arr = np.asarray(arr)
    m, n = arr.shape

    chain_means = arr.mean(axis=1)
    chain_vars  = arr.var(axis=1, ddof=1)

    W = chain_vars.mean()
    B = n * np.var(chain_means, ddof=1)
    var_hat = ((n - 1) / n) * W + (1 / n) * B

    return float(np.sqrt(var_hat / W))


def rhat_split(arr):
    """
    Split Gelman–Rubin R-hat.

    chain_stats: np.array of shape (num_chains, num_samples)

    R-hat value	Meaning
    1.00–1.05	Good mixing (usually adequate for PGM use)
    > 1.1	Chains not yet mixing; probably stuck in different energy basins
    ≫ 1.1	Very poor mixing or chains starting from very different modes
    """
    split = _split_chains(np.asarray(arr))
    return rhat_classic(split)
