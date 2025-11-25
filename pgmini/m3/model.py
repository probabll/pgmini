from pgmini.m1 import OutcomeSpace, DAG, enumerate_joint_assignments, d_separation
from pgmini.m2 import TabularFactor, UGraph 
from pgmini.m2 import separation as u_separation
from pgmini.m3 import TabularCPDFactor
from tabulate import tabulate
import functools
import itertools
import pandas as pd


class PGM:
    """A container for an MN or BN"""

    def iterrvs(self):
        """Iterate over (rv, outcome_space) pairs for the rvs in this model (in arbitrary order)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def iternodes(self):
        """Iterate over the nodes in this model (in arbitrary order)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def iteredges(self):
        """Iterate over the edges in this model (in arbitrary order)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def cardinality(self, rv):
        """The number of outcomes in the sample space of the rv"""
        raise NotImplementedError("To be implemented by specific type of model")
        
    def iterfactors(self):
        """Iterate over the factors in this model (in arbitrary order)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def enumerate_joint_assignments(self, rvs: list):
        """Enumerate joint assignments for the rvs given (in the order given)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def evaluate(self, assignment: dict):
        """Evaluate the (unnormalised) probability of the assignment. Whether this is normalised or not depends on the model family."""
        reduced_factors = [factor.reduce(assignment) for factor in self.iterfactors()]
        prod = functools.reduce(lambda a, b: a.product(b), reduced_factors)
        return prod.evaluate(dict())     

    def separate(self, X: set, Y: set, Z: set):
        """Test if X and Y are separate given Z"""
        raise NotImplementedError("To be implemented by specific type of model")


class BayesianNetwork(PGM):
    """
    A BN combines a DAG (we use an implementation from pgmini.m1)
     and a collection of CPDs (we use TabularCPD from pgmini.m1)
    """

    def __init__(self, cpd_factors: 'iterable'):
        """
        cpd_factors: a collection of TabularCPDFactor objects
            the class builds the BN structure by analysing the factors        
        """
        super().__init__()
        
        self.cpds = dict()
        self.outcome_spaces = dict()
        nodes = []
        edges = []
        for cpd in cpd_factors:
            if not isinstance(cpd, TabularCPDFactor):
                raise ValueError("BNs are parameterised by CPD factors, in this course they must be tabular.")
            self.cpds[cpd.child] = cpd
            self.outcome_spaces[cpd.child] = cpd.outcome_spaces[cpd.child]
            nodes.append(cpd.child)
            for parent in cpd.parents:
                edges.append((parent, cpd.child))        
        self.dag = DAG(nodes, edges)        

    def iterrvs(self):
        """Iterate over (rv, outcome_space) pairs for the rvs in this model (in arbitrary order)"""
        return self.outcome_spaces.items()

    def iternodes(self):
        """Iterate over the nodes in this model (in arbitrary order)"""
        return iter(self.dag.nodes)

    def iteredges(self):
        """Iterate over the edges in this model (in arbitrary order)"""
        raise iter(self.dag.edges)

    def cardinality(self, rv):
        """The number of outcomes in the sample space of the rv"""
        return len(self.outcome_spaces[rv])
        
    def iterfactors(self):
        """Iterate over the factors in this model (in arbitrary order)"""
        return iter(self.cpds.values())

    def enumerate_joint_assignments(self, rvs: list):
        return enumerate_joint_assignments(rvs, self.outcome_spaces)

    def separate(self, X: set, Y: set, Z: set):
        """Test if X and Y are separate given Z"""
        return d_separation(self.dag, X, Y, Z)
        

class MarkovNetwork(PGM):
    """
    An MN combines a UGraph (we use an implementation from pgmini.m2)
     and a collection of Factors (we use TabularCPD from pgmini.m2)
    """

    def __init__(self, factors: 'iterable'):
        """
        factors: a list of Factor objects

        We build the MN structure from the list of factors to avoid a situation where the 
        two are not coherent with one another. 
        Building it is part of an exercise, read on and you will find out more.
        """
        super().__init__()
        
        self.outcome_spaces = dict()  
        self.factors = tuple(factors)
        nodes = []
        edges = []
        for factor in self.factors:
            if not isinstance(factor, TabularFactor):
                raise ValueError("MNs are parameterised by non-negative factors, in this course they must be tabular.")
            for rv, outcome_space in factor.outcome_spaces.items():
                nodes.append(rv)                
                self.outcome_spaces[rv] = outcome_space                
            for rv1, rv2 in itertools.combinations(factor, 2):
                edges.append((rv1, rv2))        
        self.graph = UGraph(nodes, edges)
        

    def iterrvs(self):
        """Iterate over (rv, outcome_space) pairs for the rvs in this model (in arbitrary order)"""
        return self.outcome_spaces.items()

    def iternodes(self):
        """Iterate over the nodes in this model (in arbitrary order)"""
        return iter(self.graph.nodes)

    def iteredges(self):
        """Iterate over the edges in this model (in arbitrary order)"""
        raise iter(self.graph.edges)

    def cardinality(self, rv):
        """The number of outcomes in the sample space of the rv"""
        return len(self.outcome_spaces[rv])
        
    def iterfactors(self):
        """Iterate over the factors in this model (in arbitrary order)"""
        return iter(self.factors)

    def enumerate_joint_assignments(self, rvs: list):
        """Enumerate joint assignments for the rvs given (in the order given)"""
        return enumerate_joint_assignments(rvs, self.outcome_spaces)

    def separate(self, X: set, Y: set, Z: set):
        """Test if X and Y are separate given Z"""
        return u_separation(self.graph, X, Y, Z)


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

def pgm_to_tabular_factor(pgm: PGM):
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
