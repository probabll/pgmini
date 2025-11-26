import pgmini
from pgmini.m1 import OutcomeSpace
import itertools
from collections import OrderedDict
import numpy as np
from tabulate import tabulate


class Factor:
    """
    Abstract representation of a factor over a set of variables.
    """

    def __init__(self, outcome_spaces: dict):
        self.outcome_spaces = dict(outcome_spaces)

    def __contains__(self, rv: str):
        return rv in self.outcome_spaces

    def __iter__(self):
        return iter(self.outcome_spaces.keys())

    def evaluate(self, assignment: dict) -> float:
        """Evaluate the factor given assignments to all rvs in its scope"""
        raise NotImplementedError

    def reduce(self, assignment: dict) -> 'Factor':
        """
        Return a new factor with variables fixed according to the assignment dict.
        """
        raise NotImplementedError

    def marginalize(self, rvs: set) -> 'Factor':
        """
        Return a new factor with vars_to_sum_out summed out.
        """
        raise NotImplementedError

    def maximize(self, rvs: set) -> 'Factor':
        """
        Return a new factor with rvs maxed out.
        """
        raise NotImplementedError

    def product(self, other: 'Factor') -> 'Factor':
        """
        Return the product of this factor with another factor.
        """
        raise NotImplementedError

    def normalize(self) -> 'Factor':
        raise NotImplementedError


class TabularFactor(Factor):
    """
    Factor represented as a numpy array. Each variable in `scope` takes integer
    values starting from 0. The array shape corresponds to the cardinality of
    each variable.
    """

    def __init__(self, scope: list, outcome_spaces: dict, values, from_template=None):
        """
        scope: list of rv names 
            the axes of the values tensor are aligned with rvs in the same
            order as the rv names appear in this list
        outcome_spaces: dict mapping rv_name to an OutcomeSpace object for that rv
        values: numpy array with shape matching the cardinalities of the rvs
        """
        assert len(scope) == len(set(scope)), "No repetitions allowed in scope"
        super().__init__({rv_name: outcome_spaces[rv_name] for rv_name in scope})
        
        # the axes of the values tensor are ordered, 
        # so while the scope is treated as a set, this has to be treated as a sequence
        self.scope = tuple(scope)
        self.rv2axis = OrderedDict([(rv, axis) for axis, rv in enumerate(scope)])
        # tensor used to evaluate assignments of the variables in the scope
        if from_template is None:
            self.values = np.array(values, dtype=float)
        else:
            self.values = from_template

    def evaluate(self, assignment: dict):
        """Evaluate the factor given assignments to all rvs in its scope"""
        # this operation is only possible if assignment is complete, 
        # otherwise the user should be using reduce (which returns a factor)        
        outcomes = tuple(self.outcome_spaces[rv][assignment[rv]] for rv in self.scope)
        return self.values[outcomes]

    def make_slicer(self, assignment: dict):
        """
        Return a tuple to slice the tensor representation of the factor according to the assignment.
        assignment: dict mapping rv_name to its outcome string
        """
        slicer = []
        for rv in self.scope:  # only variables in scope matter
            value = assignment.get(rv, None)
            if value is None:  # if the variable is not assigned, we have nothing to index
                slicer.append(slice(None))
            else:  # if the variable is assigned, we index the respective value                
                slicer.append(self.outcome_spaces[rv][value])
        return tuple(slicer)
        
    def reduce(self, assignment: dict) -> Factor:
        """
        Fix variables according to assignment, returning a new TabularFactor with reduced scope.
        """       
        if len(self.outcome_spaces.keys() & assignment.keys()) == 0:
            return self
        slicer = self.make_slicer(assignment)
        reduced_values = self.values[slicer]
        return TabularFactor([v for v in self.scope if v not in assignment], self.outcome_spaces, reduced_values)
    
    def didactic_marginalize(self, vars_to_sum_out: set) -> Factor:
        """
        Sum out (marginalize) the listed variables from this factor, 
        returning a new TabularFactor over the remaining variables.
        """
        # the coordinates/dimensions being summed
        axes = [self.rv2axis[rv] for rv in vars_to_sum_out]
        # performs the sum axis by axis
        new_values = self.values
        for ax in sorted(axes, reverse=True):
            new_values = np.sum(new_values, axis=ax)
        new_scope = [v for v in self.scope if v not in vars_to_sum_out]
        return TabularFactor(new_scope, {rv: self.outcome_spaces[rv] for rv in new_scope}, new_values)

    def marginalize(self, vars_to_sum_out: set) -> Factor:
        """
        Sum out (marginalize) the listed variables from this factor,
        returning a new TabularFactor over the remaining variables.

        This implementation uses numpy's Einsum: 
            - it's more efficient
            - but it is not easy to read for those who are not familiar with Einsum, 
            - and it is moderately difficult to read for those who are familiar with Einsum.
        To gain a good understanding study didactic_marginalize instead.
        
        Example:
            φ(A, B, C)  --marginalize B-->  φ'(A, C)
            Einsum pattern: 'abc->ac'
        """
        # einsum cannot deal with too many axes
        if len(self.scope) > 26:            
            return self.didactic_marginalize(vars_to_sum_out)
        
        # variables to be kept after marginalisation
        remaining_vars = [v for v in self.scope if v not in vars_to_sum_out]
    
        # assign letters to variables (necessary for einsum)        
        # each variable (axis) gets a unique letter label
        var_to_letter = {v: chr(97 + i) for i, v in enumerate(self.scope)}
        # e.g., {'A': 'a', 'B': 'b', 'C': 'c'}
    
        # build the einsum expression        
        # input subscripts: one letter per variable in this factor
        idx_in = ''.join(var_to_letter[v] for v in self.scope)
        # output subscripts: same but dropping marginalized variables
        idx_out = ''.join(var_to_letter[v] for v in remaining_vars)
        # Example: 'abc->ac' (summing over 'b')
        expr = f"{idx_in}->{idx_out}"
            
        # This automatically sums out all omitted indices
        new_values = np.einsum(expr, self.values)
            
        return TabularFactor(remaining_vars, {rv: self.outcome_spaces[rv] for rv in remaining_vars}, new_values)

    def maximize(self, rv: str) -> Factor:
        """    
        Max one rv out of the factor, returning a new TabularFactor over the remaining variables.

        Example:
            φ(A, B, C)  --maximize B-->  φ'(A, C) = \max_b φ(A, B=b, C)

        rv: the rv to be maximized away.
            If the rv is not relevant to this factor, we simply return the factor itself.
        """        
        if rv not in self:
            return self
        axis = self.rv2axis[rv]
        new_values = np.max(self.values, axis=axis)
        remaining_vars = [v for v in self.scope if v != rv]        
        return TabularFactor(remaining_vars, {v: self.outcome_spaces[v] for v in remaining_vars}, new_values)


    def _argmax_rv(self) -> str:
        """
        Return the argmax of the value tensor.
        This method only works when the factor has a single rv in its scope.
        (This method provides auxiliary functionality to other methods in this class, 
         user code wil seldom need to call this)
        """        
        assert len(self.scope) == 1, "This primitive only works for factors over a single rv"                
        rv = self.scope[0]
        axis = self.rv2axis[rv]
        argmax = np.argmax(self.values, axis=axis)
        return self.outcome_spaces[rv].outcomes[argmax] 

    def argmax_rv(self, query_rv: str, evidence: dict) -> str:
        """
        For a factor phi with scope (Q, E), where Q is a query rv and E is a set of evidence rvs,             
            return the outcome of Q that maximizes phi[E=e](Q).         
        That is:
            argmax_{q in Val(Q)} phi[E=e](Q).

        Withou Q and E, we return the argmax across the entire table.
        
        query_rv: the rv (in scope) for which we want the argmax.
        evidence: an assignment of the remaining rvs in the factor's scope
            this function requires assigning all rvs except Q.
        """          
        f = self.reduce(evidence)
        return f._argmax_rv()        
    
    def argmax(self):
        """
        For a factor phi with scope (Q, E), where Q is a query rv and E is a set of evidence rvs,             
            return the outcome of Q that maximizes phi[E=e](Q).         
        That is:
            argmax_{q in Val(Q)} phi[E=e](Q).

        Withou Q and E, we return the argmax across the entire table.
        
        query_rv: the rv (in scope) for which we want the argmax.
        evidence: an assignment of the remaining rvs in the factor's scope
            this function requires assigning all rvs except Q.
        """        
        flat_argmax = self.values.flatten().argmax()
        struct_argmax = np.unravel_index(flat_argmax, self.values.shape)
        return {rv: self.outcome_spaces[rv].outcomes[argmax] for rv, argmax in zip(self.scope, struct_argmax)}
        
    def _sample_rv(self, rng) -> str:
        """
        Return an sampled outcome. 
        This method only works when the factor has a single rv in its scope.
        (This method provides auxiliary functionality to other methods in this class, 
         user code wil seldom need to call this)
        """
        assert len(self.scope) == 1, "This primitive only works for factors over a single rv"                
        rv = self.scope[0]
        idx = rng.choice(len(self.outcome_spaces[rv]), p=self.values / np.sum(self.values))
        return self.outcome_spaces[rv].outcomes[idx]

    def sample_rv(self, query_rv: str, evidence: dict, rng=np.random.default_rng()) -> str:
        """
        Sample the outcome of one RV given all other RVs. 
        query_rv: the rv (in scope) to be sampled
        evidence: an assignment of all other rvs (irrelevant rvs are ignored)
        rng: an np random number generator
        """
        f = self.reduce(evidence)
        return f._sample_rv(rng)
    
    def sample(self, size=None, rng=np.random.default_rng()):
        """
        Sample one or more assignments of all rvs from the normalized version of the factor.
        size: as in numpy
            - if size is None, one sampled assignment (a dict) will be returned;
            - otherwise, a list containining `size` sampled assignments (each a dict) will be returned
            (note that this means that if size is 1, then a list with 1 assignment will be returned)
        """
        # flatten and normalise the table 
        p = self.values.flatten() / np.sum(self.values)
        if size is None:
            # draw a sample
            flat_sample = rng.choice(len(p), p=p)
            # find the structured ids of the outcomes in the original tensor
            struct_sample = np.unravel_index(flat_sample, self.values.shape)
            # make a dict assignment
            return {rv: self.outcome_spaces[rv].outcomes[idx] for rv, idx in zip(self.scope, struct_sample)}
        else:
            # draw as many samples as requested
            flat_samples = rng.choice(len(p), size=size, p=p)
            samples = []
            for flat_sample in flat_samples:  # for each flat sample
                # find the structured ids of the outcomes in the original tensor
                struct_sample = np.unravel_index(flat_sample, self.values.shape)
                # make a dict assignment
                sample = {rv: self.outcome_spaces[rv].outcomes[idx] for rv, idx in zip(self.scope, struct_sample)}    
                samples.append(sample)
            return samples

    def didactic_product(self, other) -> Factor:
        """
        Multiply two tabular factors returning a new TabularFactor object with 
        a new scope. When factors' scopes overlap, we figure out the correct aligned scope.

        This implementation explicitly enumerates the joint outcome spaces of the product factor, 
         hence it can be inefficient if we have too large scopes.
        """
        # determine combined scope
        new_scope = list(OrderedDict.fromkeys(self.scope + other.scope))
        # merge outcome spaces dicts
        new_spaces = {**self.outcome_spaces, **other.outcome_spaces}
        # shape of the new tensor
        shape = [len(new_spaces[v]) for v in new_scope]
        new_values = np.zeros(shape)
        # iterate over joint assignments
        # fetch the relevants values from each factor
        # multiply values
        for assignment in itertools.product(*(range(len(new_spaces[v])) for v in new_scope)):
            a_dict = dict(zip(new_scope, assignment))
            val1 = self.values[tuple(a_dict[rv] for rv in self.scope)]
            val2 = other.values[tuple(a_dict[rv] for rv in other.scope)]
            new_values[assignment] = val1 * val2
        return TabularFactor(new_scope, new_spaces, new_values)

    def product(self, other) -> Factor:
        """
        Multiply two tabular factors returning a new TabularFactor object with 
        a new scope. When factors' scopes overlap, we figure out the correct aligned scope.

        This implementation uses numpy's Einsum: 
            - it's more efficient
            - but it is not easy to read for those who are not familiar with Einsum, 
            - and it is moderately difficult to read for those who are familiar with Einsum.
        To gain a good understanding study didactic_product instead.
        
        Example:
            φ1(A, B) × φ2(B, C)  →  φ3(A, B, C)
            Einsum pattern: 'ab,bc->abc'        
        """        
        # combined scope 
        all_vars = list(OrderedDict.fromkeys(self.scope + other.scope))
        # einsum cannot deal with too many axes
        if len(all_vars) > 26:
            # the didactic version can (in principle) but it is going to be slow
            return self.didactic_product(other)
        
        # Construct the einsum pattern for the operation
    
        # First, map each variable to a single character
        # so that einsum can identify which axes correspond to which variables
        var_to_letter = {v: chr(97 + i) for i, v in enumerate(all_vars)}

        # Then construct the pattern
        # For φ1(A,B) → 'ab'
        idx_self = ''.join(var_to_letter[v] for v in self.scope)
        # For φ2(B,C) → 'bc'
        idx_other = ''.join(var_to_letter[v] for v in other.scope)
        # For output over (A,B,C) → 'abc'
        idx_out = ''.join(var_to_letter[v] for v in all_vars)
        # Combine to form Einstein summation expression
        expr = f"{idx_self},{idx_other}->{idx_out}"
        # Example: 'ab,bc->abc'
            
        # einsum automatically:
        #   - multiplies arrays elementwise where indices match
        #   - sums over any repeated indices (shared variables)
        # So 'ab,bc->abc' implements sum_b φ1(a,b)*φ2(b,c)
        new_values = np.einsum(expr, self.values, other.values)
    
        return TabularFactor(all_vars, {**self.outcome_spaces, **other.outcome_spaces}, new_values)

    def normalize(self) -> Factor:
        """
        Return a globally normalized factor (same as marginalize(self.scope) but implemented more directly).
        """
        Z = np.sum(self.values)
        if Z > 0:            
            return TabularFactor(self.scope, self.outcome_spaces, self.values / Z)
        else:
            raise ValueError(f"I need Z > 0, got Z={Z}")

    def display(self, tablefmt="markdown", factor_name="Value"):
        """Render the TabularFactor as a string for visualisation using tabulate"""
        data = []        
        scope = list(self.scope)
        joint_outcomes = OutcomeSpace.enumerate_joint_outcomes(*(self.outcome_spaces[rv] for rv in self.scope))        
        for outcomes, value in zip(joint_outcomes, self.values.flatten()):
            data.append(list(outcomes) + [value])
        return tabulate(data, headers=scope + [factor_name], tablefmt=tablefmt)

    def __str__(self):
        return self.display()

    def __repr__(self):
        return str(self)

