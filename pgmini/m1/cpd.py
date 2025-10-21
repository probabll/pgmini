from collections import OrderedDict
import itertools
import numpy as np
from tabulate import tabulate

class OutcomeSpace:
    """
    A container to hold a countably finite set of outcome objects.
    This container maps an outcome to a unique 0-based identifier (id), 
     this way we can use an np.array to store a dense representation of a tabular cpds;
     each axis is associated with an outcome space and the unique id corresponds
     to the index of the outcome object along that axis. 
     

    """
    
    def __init__(self, outcomes):
        """
        outcomes: an iterable of objects that can be hashed
        """
        # Ensure outcomes are unique and preserve order
        self.outcomes = tuple(dict.fromkeys(outcomes))
        # Map each outcome to a unique 0-based id
        self._outcome2id = {outcome: i for i, outcome in enumerate(self.outcomes)}

    def __len__(self):
        """Return the number of outcomes"""
        return len(self.outcomes)

    def __iter__(self):
        """Iterate over outcomes"""
        return iter(self.outcomes)

    def __contains__(self, outcome):
        """Check if an outcome is a member of the outcome space"""
        return outcome in self._outcome2id

    def __getitem__(self, outcome):
        """Get the id corresponding to an outcome"""
        return self._outcome2id[outcome]

    def __repr__(self):
        return f"OutcomeSpace({len(self)} outcomes)"

    def __str__(self):
        return "{%s}" % ', '.join(str(o) for o in self.outcomes)
        
    @classmethod
    def enumerate_joint_outcomes(cls, *spaces):
        """
        Return a generator for joint outcomes in the product space of the given spaces.
        *spaces: an arbitrary number of iterables
          e.g., enumerate_joint_outcomes([1, 2, 3], ('a', 'b'), iter([False, True]))
          yields elements in the cross product space of these 3 iterables
        """
        for joint_outcome in itertools.product(*spaces):
            yield joint_outcome


class TabularCPD:
    """
    A container to hold a conditional probability distribution (CPD) 
    represented as a table (a dense numpy array). 

    It uses OutcomeSpaces to map named categories to 0-based integers, so we can index the table efficiently.

    This version always has a single child rv (CPDs in general can have multiple children rvs, 
    but for BNs such a feature is not really necessary). The list of parents may be empty or contain multiple rvs.

    The order of the parents in the list has no semantics in a CPD, 
    but in its _tabular_ representation it tells us which axis of the underlying table captures
    which rv's outcome space. Hence, the order matters for the implementation. 
    Any one parent should be listed only once.
    """

    def __init__(self, parents, child, outcome_spaces: dict, table, tol=1e-6):
        """
        parents: names of the parent rvs, each a string
            no duplicates, the order of parents in this list tells the implementation
            which axis of the table is to be associated with which rv's outcome space
        child: the name of the child rv (string)
        outcome_spaces: dict mapping an rv name to its OutcomeSpace
            parents and child must be in the dict
            (this implementation ignores any irrelevant space that may be in the dict)
        table: np.array containing all probabilities, the table's shape is given by 
            the cardinalities of the parents (in order), 
            followed by the cardinality of the child,
            where cardinality is the size of the rv's outcome space 

            the probabilities should add up to 1.0 over the outcome space of the child rv
            for any joint assignment of the parents.
            
        tol: a tolerance parameter when checking whether distributions add to 1.0
        """
        self.parents = tuple(parents)
        self.child = child
        self.outcome_spaces = OrderedDict((v, outcome_spaces[v]) for v in self.parents + (child,))
        shape = tuple(len(space) for space in self.outcome_spaces.values())        
        self.table = np.array(table)                
        assert self.table.shape == shape, f"I need a table of shape {shape} but got {self.table.shape}"
        assert np.allclose(np.sum(self.table, -1), 1, tol), "Some CPDs are not summing to 1.0"
    
    def enumerate_outcomes(self):
        """Iterable for outcomes of the _child_ rv"""
        return iter(self.outcome_spaces[self.child])

    def enumerate_assignments(self):
        """Iterable for assignment dict of the _child_ rv"""
        return ({self.child: outcome} for outcome in self.outcome_spaces[self.child])

    def prob(self, assignment: dict):
        """
        Return probability of the assignment, where an assignment is a dictionary mapping 
        an rv (by its name) to an outcome (a named category).
        The assignment dict is such that:
            * parents must be assigned
            * child must be assigned
        Irrelevant rvs in the dict are ignored by this method.
        """
        # map parents' outcomes to ids in the order we established for the axes of the np.array
        ctxt_idx = tuple(self.outcome_spaces[par_rv][assignment[par_rv]] for par_rv in self.parents)
        # map child's outcome to id
        idx = self.outcome_spaces[self.child][assignment[self.child]]
        return self.table[ctxt_idx][idx]

    def __str__(self):
        """Render the CPD as a string for visualisation using tabulate"""
        data = []        
        par_names = list(self.parents)
        outcome_space = list(f"{self.child}={outcome}" for outcome in self.outcome_spaces[self.child])
        joint_ctxt_outcomes = OutcomeSpace.enumerate_joint_outcomes(*(self.outcome_spaces[par_var] for par_var in self.parents))
        all_cpds = self.table.reshape(-1, len(self.outcome_spaces[self.child]))
        for ctxt, cpd in zip(joint_ctxt_outcomes, all_cpds):
            data.append(list(ctxt) + cpd.tolist())
        return tabulate(data, headers=par_names + outcome_space, tablefmt="markdown")

    def __repr__(self):
        return str(self)