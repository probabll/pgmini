from pgmini.m1 import OutcomeSpace, DAG
from pgmini.m2 import TabularFactor, UGraph
from pgmini.m3 import TabularCPDFactor
import functools
import itertools


class PGM:
    """A container for an MN or BN"""

    def iterrvs(self):
        """Iterate over the rvs in this model (in arbitrary order)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def cardinality(self, rv):
        """The number of outcomes in the sample space of the rv"""
        raise NotImplementedError("To be implemented by specific type of model")
        
    def iterfactors(self):
        """Iterate over the factors in this model (in arbitrary order)"""
        raise NotImplementedError("To be implemented by specific type of model")

    def enumerate_joint_assignments(self, rvs):
        raise NotImplementedError("To be implemented by specific type of model")

    def evaluate(self, assignment: dict):
        reduced_factors = [factor.reduce(assignment) for factor in self.iterfactors()]
        prod = functools.reduce(lambda a, b: a.product(b), reduced_factors)
        return prod.evaluate(dict())     


class BayesianNetwork(PGM):
    """
    A BN combines a DAG (we use an implementation from pgmini.m1)
     and a collection of CPDs (we use TabularCPD from pgmini.m1)
    """

    def __init__(self, cpd_factors):
        """
        cpd_factors: a collection of TabularCPDFactor objects
            the class builds the BN structure by analysing the factors        
        """
        super().__init__()
        nodes = []
        edges = []
        self.cpds = dict()
        self.outcome_spaces = dict()
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
        return iter(self.dag.nodes)

    def cardinality(self, rv):
        return len(self.outcome_spaces[rv])
        
    def iterfactors(self):
        return iter(self.cpds.values())

    def enumerate_joint_assignments(self, rvs):
        return enumerate_joint_assignments(rvs, self.outcome_spaces)

class MarkovNetwork(PGM):
    """
    An MN combines a UGraph (we use an implementation from pgmini.m2)
     and a collection of Factors (we use TabularCPD from pgmini.m2)
    """

    def __init__(self, factors: list):
        """
        factors: a list of Factor objects

        We build the MN structure from the list of factors to avoid a situation where the 
        two are not coherent with one another. 
        Building it is part of an exercise, read on and you will find out more.
        """
        super().__init__()
        nodes = []
        self.outcome_spaces = dict()
        for factor in factors:
            if not isinstance(factor, TabularFactor):
                raise ValueError("MNs are parameterised by non-negative factors, in this course they must be tabular.")
            for rv, outcome_space in factor.outcome_spaces.items():
                nodes.append(rv)                
                self.outcome_spaces[rv] = outcome_space
        edges = []
        for factor in factors:
            for rv1, rv2 in itertools.combinations(factor, 2):
                edges.append((rv1, rv2))
        
        self.graph = UGraph(nodes, edges)
        self.factors = list(factors)

    def iterrvs(self):
        return iter(self.graph.nodes)

    def cardinality(self, rv):
        return len(self.outcome_spaces[rv])
        
    def iterfactors(self):
        return iter(self.factors)

    def enumerate_joint_assignments(self, rvs):
        return enumerate_joint_assignments(rvs, self.outcome_spaces)
