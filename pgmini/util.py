import pandas as pd
import numpy as np
import functools
from tabulate import tabulate
from pgmini.m2 import TabularFactor
from pgmini.m3 import PGM


def display_full_table(pgm: PGM, rvs=None, normalize=False, tablefmt='simple'):
    """
    Return a tabulate-formatted string that can be printed to display the whole (unnormalised) joint table.
    
    pgm: an instance of PGM
    rvs: optionally specify the order in which to list rvs in the table
    normalize: whether or not the output is to be normalized
    """
    table = []
    if rvs is None:
        rvs = list(pgm.iternodes())
    total = 0.0
    for assignment in pgm.enumerate_joint_assignments(rvs):
        value = pgm.evaluate(assignment)
        total += value
        table.append([assignment[rv] for rv in rvs] + [value])
    if normalize and total > 0.:
        for row in table:
            row[-1] /= total
        return tabulate(table, headers=rvs + ['P'], tablefmt=tablefmt)
    else:
        return tabulate(table, headers=rvs + ['~P'], tablefmt=tablefmt)


def pgm_to_tabular_factor(pgm: PGM) -> TabularFactor:
    """Compute the product of all factors in the PGM returning a large TabularFactor"""
    return functools.reduce(lambda a, b: a.product(b), pgm.iterfactors())
    
    
def pgm_to_df(pgm: PGM, rvs=None, normalize=False, Z=None):
    """
    Return a pandas DataFrame containing a complete table-view of the joint unnormalised distribution represented by a PGM.
    
    pgm: an instance of PGM
    rvs: optionally specify the order in which to list rvs in the table
    normalize: whether or not the output is to be normalized
    Z: the normalizer of the pgm
        (either use normalize or give the normalizer Z, not both)
    """
    assert not (normalize and Z is not None), "Provide a normalization constant Z or request normalization, not both"
        
    table = []
    if rvs is None:
        rvs = list(pgm.iternodes())
    total = 0.0
    if Z is None:
        Z = 1.0    
    for assignment in pgm.enumerate_joint_assignments(rvs):
        value = pgm.evaluate(assignment)
        total += value
        table.append([assignment[rv] for rv in rvs] + [value / Z])
    if normalize and total > 0.:
        for row in table:
            row[-1] /= total
    return pd.DataFrame(table, columns=rvs + ['Value'])



def make_samples_df(samples: list, count_col="Count", prob_col="Value"):
    """
    Return a pandas DataFrame where each row is an assignment its count and relative frequency in the sample.
    samples: a list of assignments (each a dict)
    """
    
    # convenient data frame with counts and relative frequency of outcomes
    samples_df = pd.DataFrame(samples)
    counts = samples_df.value_counts().reset_index(name=count_col)
    
    # Add relative frequency column
    counts[prob_col] = counts[count_col] / counts[count_col].sum()
    return counts


def df_to_factor(df, rvs: list, outcome_spaces: dict, value_col="Value", miss_value=0) -> TabularFactor:
    """
    Convert a DataFrame of TabularFactor.
    Note that this makes the representation dense (missing assignments in the DataFrame will have 0 score in the TabularFactor)
    
    pgm: used to determined the rvs and their outcome spaces
    df: pandas.DataFrame
        containing sampled outcomes, their counts and frequency    
    prob_col : str
      Name of the column containing estimated probabilities    
    """    
    rvs = list(rvs)
    # Determine number of states per RV
    shape = [len(outcome_spaces[var]) for var in rvs]

    # Initialize tensor
    tensor = np.zeros(shape, dtype=float) + miss_value

    # Fill tensor
    for _, row in df.iterrows():
        # Get indices in tensor according to rv_order
        idx = tuple(outcome_spaces[var][row[var]] for var in rvs)
        tensor[idx] = row[value_col]

    return TabularFactor(rvs, {rv: outcome_spaces[rv] for rv in rvs}, tensor)    


def tvd(p, q, prob_col="Value"):
    """
    The total variation distance (TVD) between two discrete distributions P and Q over the same rvs X is defined as 
        1/2 \sum_{x\in Val(X)} |P(X=x) - Q(X=x)|

    p and q can be 
    - two np.ndarray objects
    - two normalised TabularFactor objects 
    - two pd.DataFrame objects (from make_samples_df)
    """
    if isinstance(p, np.ndarray) and isinstance(q, np.ndarray):
        return 0.5 * np.abs(p - q).sum()
    elif isinstance(p, TabularFactor) and isinstance(q, TabularFactor):        
        perm = [q.scope.index(rv) for rv in p.scope]
        return 0.5 * np.abs(p.values - q.values.transpose(perm)).sum()
    elif isinstance(p, pd.DataFrame) and isinstance(q, pd.DataFrame):
        # All RV columns (everything except the probability column)
        rv_cols = [c for c in p.columns if c != prob_col]
    
        # Merge on the RV assignments
        merged = p.merge(q, on=rv_cols, how="outer", suffixes=("_p", "_q"))
    
        # Missing probabilities → 0
        merged[f"{prob_col}_p"] = merged[f"{prob_col}_p"].fillna(0)
        merged[f"{prob_col}_q"] = merged[f"{prob_col}_q"].fillna(0)
    
        # Total variation distance
        return 0.5 * (merged[f"{prob_col}_p"] - merged[f"{prob_col}_q"]).abs().sum()
    else:
        raise NotImplementedError("I need np.ndarray objects, or TabularFactor objects, or pd.DataFrame objects")

        