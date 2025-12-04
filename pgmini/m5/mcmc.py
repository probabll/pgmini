from pgmini.m3 import PGM
import numpy as np
from pgmini.m4 import split_factors
import functools
from joblib import Parallel, delayed


def draw_initial_assignment(pgm: PGM, rng=np.random.default_rng()):
    """
    Obtain an initial assignment of the PGM's RVs by any one of the procedures explained above.

    pgm: the PGM we are drawing the initial assignment from
    rng: a numpy random number generator

    The return type is a dictionary representing the assignment.
    """

    assignment = dict()
    for X, Val_X in pgm.iterrvs():
        id_x = rng.choice(len(Val_X))
        assignment[X] = Val_X.outcomes[id_x]
    return assignment


def gibbs_sampler(
    pgm: PGM,   
    initial_assignment: dict,
    num_iterations: int,
    burn_in: int = 0,
    thinning: int = 1,
    order=None,
    shuffle_order=True,
    rng=np.random.default_rng(),
):   
    """
    Simulate a Gibbs chain for a number of iterations and return a list of samples (each a dict).
    (possibly after discarding some samples, depending on burn-in and thinning).
    
    pgm: the model we are sampling from
        (if you need to condition on something, construct a PGM with reduced factors before calling this function)
    initial_assignment: a complete assignment of all of the pgm's rvs
    num_iterations: how many iterations we run the Gibbs sampler for
    burn_in: how many samples we discard from the beginning of the chain
    thinning: for some number k, collect every kth sample, discard the rest
    order: in which order should we resample rvs
    shuffle_order: whether to shuffle the order (uniformly at random) at the beginning of each iteration
    rng: a numpy random number generator
        (necessary so we can sample from factors; if not provided, we use np.random.default_rng())
    """    
    variables = set(pgm.iternodes())
    # validate
    assert variables <= set(initial_assignment.keys()), "initial_assignment must define every variable"
    
    if order is None: # fix an order if none is given
        order = list(pgm.iternodes())

    current_assignment = dict(initial_assignment)  
    
    samples = []
    for j in range(num_iterations):
        if shuffle_order:  # shuffle the order if needed
            permutation = rng.permutation(len(order))
        else:
            permutation = np.arange(len(order))
        for i in permutation:  # for each rv in order
            X = order[i]
            del current_assignment[X]  # remove rv from assignment
            # find relevant factors
            relevant, irrelevant = split_factors(X, pgm.iterfactors())
            # reduce the relevant factors (this is fixing the rvs in the rv's Markov Blanket)
            relevant = [f.reduce(current_assignment) for f in relevant]
            
            # compute product of factor for all relevant, reduced factors
            prod = functools.reduce(lambda a, b: a.product(b), relevant)
            # normalisation gives us the probability distribution over X given MB(X)
            P_X = prod.normalize()  
                        
            # then we can sample an new outcome
            # and that gives us a new assignment
            current_assignment[X] = P_X.sample(rng=rng)[X]
            
        # at the end of a complete pass through all rvs, we have a sample
        samples.append(dict(current_assignment))

    return samples[burn_in:][::thinning]


def run_chain(sampler_fn, seed, init, args, kwargs):
    """Run one chain calling sampler_fn"""
    rng = np.random.default_rng(seed)
    return sampler_fn(*args, rng=rng, **kwargs, initial_assignment=init)

def run_chains_joblib(sampler_fn, inits: list, *args, **kwargs):
    """Run K chains in paralell, using sampler_fn, we determine the number K by checking the size of the inits list."""
    K = len(inits)
    master_rng = np.random.default_rng()
    seeds = master_rng.integers(0, 2**63-1, size=K)
    
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_chain)(sampler_fn, seed, init, args, kwargs)
        for seed, init in zip(seeds, inits)
    )
    return np.stack(results)