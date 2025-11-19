from pgmini.m2 import TabularFactor
import numpy as np


class TabularCPDFactor(TabularFactor):
    """
    This is just a TabularCPD but it is implemented via a TabularFactor
    """

    def __init__(self, parents: tuple, child: str, outcome_spaces: dict, values, tol=1e-6):
        """
        parents: list of parent rvs
        child: child rv            
        outcome_spaces: dict mapping rv_name to an OutcomeSpace object for that rv
            it should contain the parents and the child, anything else will be ignored
        values: numpy array with shape matching the cardinalities of the rvs
            the axes of the values tensor are aligned with rvs in the 
            order: tuple(parents) + (child,)
        """
        super().__init__(tuple(parents) + (child,), outcome_spaces, values)
        self.child = child
        self.parents = tuple(parents)
        assert np.allclose(np.sum(self.values, -1), 1, tol), "Some CPDs are not summing to 1.0"