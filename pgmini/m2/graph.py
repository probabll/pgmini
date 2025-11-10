import itertools
from collections import defaultdict, deque
from tabulate import tabulate

class UGraph:
    """
    A container for an undirected graph.

    The container holds nodes that are string objects 
    (or, at least, that overwrite the __str__ method).

    We store a tuple of nodes and a tuple of edges.
    For convenience of various graph algs, we also store adjacency maps.
    """

    def __init__(self, nodes: list, edges: list):
        """
        nodes: a list of string objects (or objects supporting str(obj))
        edges: a list of edges, each represented as a (parent, child) pair
        """
        self.nodes = frozenset(str(node) for node in nodes)
        self.edges = frozenset((str(a), str(b)) if str(a) < str(b) else (str(b), str(a)) for a, b in edges)        
        self.neighbors = defaultdict(set)
        for u, v in self.edges:
            self.neighbors[u].add(v)
            self.neighbors[v].add(u)
        self.neighbors = {u: frozenset(vs) for u, vs in self.neighbors.items()}

    def __hash__(self):
        return hash((self.nodes, self.edges))

    def markov_blanket(self, node):
        """Return the neighbours of a node"""
        return self.neighbors.get(node, frozenset())

    def enumerate_paths(self, start, end, visited=None):
        """
        Return a generator for all paths between two nodes (arbitrarily called start and end).
        
        start: a node
        end: another node
        visited: a set of visited nodes (used in recursive calls)
        
        Note that enumerating 1 path is always tractable (scales with the size of the graph), 
        but enumerating all paths between two nodes is in general intractable 
        (worst case is exponential in the number of nodes in the graph). 
        Using this for some visualisations is fine, but for algorithms like sep and such
        this isn't the best way to do things. 
        """
        if visited is None:
            visited = set()
        visited.add(start)
        if start == end:
            yield [start]
        else:
            neighbors = self.neighbors[start]
            for nb in neighbors:
                if nb not in visited:
                    for path in self.enumerate_paths(nb, end, visited.copy()):
                        yield [start] + path

    
    def __str__(self):
        """Generate a view of the graph using tabulate"""
        rows = []
        for u, v in sorted(self.edges):
            rows.append([f"{u} -- {v}"])
        return tabulate(rows, headers=['edges'], tablefmt='grid')

    def __repr__(self):
        return str(self)


def separation(graph: UGraph, X: set, Y: set, Z: set):
    """
    Return True if sep(X;Y|Z), and False otherwise.
    """    
    X, Y, Z = set(X), set(Y), set(Z)

    # Early termination: if any X or Y are in Z, they’re trivially separated
    if X & Z or Y & Z:
        return True

    # BFS starting from all nodes in X
    visited = set()
    queue = deque(X)

    while queue:
        node = queue.popleft()

        if node in visited:
            continue
        visited.add(node)

        # If we reach any Y node → not separated
        if node in Y:
            return False

        # Add all neighbors except those "blocked" by Z
        for nb in graph.neighbors.get(node, set()):
            if nb not in visited and nb not in Z:
                queue.append(nb)

    # No active path from X to Y given Z → separated
    return True
