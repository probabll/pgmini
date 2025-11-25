from pgmini.m3 import PGM, BayesianNetwork, MarkovNetwork
import functools


def naive_exact_inference(query_rvs: set, evidence: dict, pgm: PGM, reduce_first=False):
    """
    Return the (unnormalized) distribution over Q given E=e.
    
    In naive exact inference we build the complete joint factor before conditioning
    on evidence, then marginalise the necessary variables.

    A first optimisation we can consider is to reduce to evidence first, and only then
    multiply all factors together.

    query_rvs: the set Q 
    evidence: a dict representing the evidence assignment E=e
        Q and E are disjoint
    pgm: a PGM (a BN or an MN)    
    reduce_first: an optimisation whereby we reduce the factors to evidence
        before taking their product
    normalize: whether or not we normalize the result.
    """
    assert len(set(query_rvs) & evidence.keys()) == 0, "Q and E should be disjoint"    
    
    if reduce_first:
        # reduce all factors to evidence
        factors = [factor.reduce(evidence) for factor in pgm.iterfactors()]
        out = functools.reduce(lambda a, b: a.product(b), factors)
    else:
        out = functools.reduce(lambda a, b: a.product(b), pgm.iterfactors())
        out = out.reduce(evidence)
    
    all_rvs = set(pgm.iternodes())
    rvs_to_marginalize = all_rvs - set(query_rvs) - evidence.keys()
    
    for rv in rvs_to_marginalize:
        out = out.marginalize(rv)

    return out


def split_factors(rv, all_factors):
    """
    Splits all_factors into a list that's relevant to the rv and another that's irrelevant.
    A factor is "relevant" to an rv if that rv is in the factor's scope.
    """
    relevant, irrelevant = [], []
    for factor in all_factors:
        if rv in factor:
            relevant.append(factor)
        else:
            irrelevant.append(factor)
    return relevant, irrelevant

def sum_product_variable_elimination(pgm: PGM, query_rvs=set(), evidence=dict(), key=None, sep_opt=True, trace=None):
    """
    For a PGM with nodes X, and with two disjoint subsets of X, namely, Q and E.
    
    Return a factor representation of the (unnormalised) distribution over Q given E=e.
    That is, 
        P(Q, E=e) = \sum_W P(W, Q, E=e)
        where W is the complement of union(Q, E) in X. 

    pgm: the model we are performing VE for
    query_rvs: the set Q 
    evidence: a dict representing the evidence assignment E=e
        Q and E are disjoint
    key: use to specify the order of elimination 
        (this is passed to python's sorted function)
    sep_opt: optimisation using graphical separation
    trace: if provided, we log here each variable that was eliminated, in order, and the scope
        of the factor from which we eliminated it
    """
    assert len(set(query_rvs) & evidence.keys()) == 0, "Q and E should be disjoint"    

    if isinstance(pgm, BayesianNetwork):  # we moralise the BN
        pgm = MarkovNetwork(pgm.iterfactors())
    
    # reduce all factors using the available evidence
    factors = [factor.reduce(evidence) for factor in pgm.iterfactors()]
    # rvs that need to be eliminated/marginalised
    all_rvs = set(pgm.iternodes())
    Q = set(query_rvs)
    E = set(evidence.keys())
    rvs_to_marginalize = all_rvs - Q - E

    # eliminate each rv in order
    for rv in sorted(rvs_to_marginalize, key=key):
        # separate factors containing this rv in their scope
        involved, factors = split_factors(rv, factors)
        if not involved:  # nothing to do here
            if trace is not None:  # possibly log what we did
                trace.append((rv, tuple()))
            continue

        # when rv is separate from Q given E, we have nothing to do
        # by calling split_factors before this test
        # we are sure that the factors that concern this separate rv have been removed
        # from the collection    
        if sep_opt and pgm.separate(X={rv}, Y=Q, Z=E):
            if trace is not None:  # possibly log what we did
                trace.append((rv, tuple()))
            continue
        
        # take the product of the factors involved in this elimination step
        new_factor = functools.reduce(lambda a, b: a.product(b), involved)        
        if trace is not None:  # possibly log what we did
            trace.append((rv, new_factor.scope))
        # marginalise / eliminate the rv from the factor product
        new_factor = new_factor.marginalize({rv})
        # keep the factor for future use
        factors.append(new_factor)
    
    # whatever factors remain can parameterise an MN for the query 
    return MarkovNetwork(factors)


def max_product_variable_elimination(pgm: PGM, evidence=dict(), key=None, trace=None):
    """
    Let X be all rvs in the PGM, and E=e be some evidence (E is a subset of X), then 
        W is the complement of E in X (that is, W is the rvs in X for which we have no evidence). 
        
    This function returns the assignment of W which has highest probability under P(W|E=e)
     as well as the MN representation that remains after running max-product VE. 
     
    This is equivalent to solving 
        w* = argmax_W ~P(W, E=e)
    which is what this function does. To avoid building the complete tabular view of P(X), 
    this function uses variable elimination in a given order. 

    pgm: the model we are performing VE for
    evidence: a dict representing the evidence assignment E=e 
    key: use to specify the order of elimination 
        (this is passed to python's sorted function)
    trace: if provided, we log here each variable that was eliminated, in order, and the scope
        of the factor from which we eliminated it
    """

    if isinstance(pgm, BayesianNetwork):  # we moralise the BN
        pgm = MarkovNetwork(pgm.iterfactors())
    
    # reduce all factors using the available evidence
    factors = [factor.reduce(evidence) for factor in pgm.iterfactors()]
    # rvs that need to be eliminated/marginalised
    all_rvs = set(pgm.iternodes())
    rvs_to_maximize = all_rvs - evidence.keys()

    # this trace is not for logging, it is an integral part of the algorithm
    # it helps us recover the argmax assignment at the end
    max_trace = []
    # eliminate each rv in order
    for rv in sorted(rvs_to_maximize, key=key):
        # separate factors containing this rv in their scope
        involved, factors = split_factors(rv, factors)
        if not involved:  # nothing to do here
            if trace is not None:  # possibly log what we did
                trace.append((rv, tuple()))
            continue
        # take the product of the factors involved in this elimination step
        inter_factor = functools.reduce(lambda a, b: a.product(b), involved)        
        # keep the intermediate factor for later use (in building the argmax assignment)
        max_trace.append((rv, inter_factor))
        if trace is not None:  # possibly log what we did
            trace.append((rv, inter_factor.scope))
        # maximise / eliminate the rv from the factor product
        new_factor = inter_factor.maximize(rv)        
        # keep the factor for future use
        factors.append(new_factor)

    # build the argmax assignment
    argmax_assignment = dict()
    # we traverse the trace in reverse, so we know we have
    # solved the argmax assignment of any variable that might be needed at any one point
    for rv, f in reversed(max_trace):  
        # we assign all rvs we can, and then get the argmax for the rv that remains in the factor's scope
        outcome = f.argmax(query_rv=rv, evidence=argmax_assignment)
        argmax_assignment[rv] = outcome        

    return argmax_assignment, MarkovNetwork(factors)