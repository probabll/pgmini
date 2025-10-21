from collections import defaultdict, deque
from tabulate import tabulate

def build_adjacency(nodes: list, edges: list):
    """
    Build and return adjacency maps, one for children and one for parents.
    """
    children = {n: set() for n in nodes}
    parents = {n: set() for n in nodes}
    for u, v in edges:
        children[u].add(v)
        parents[v].add(u)    
    return children, parents


def topological_sort(nodes: list, children: dict, parents: dict):
    """Return nodes in a topological order"""
    # number of incoming edges to a node,
    # we will use this to track if how many of a node's parents
    # we have already managed to put in order
    indegree = {n: len(parents[n]) for n in nodes}
    # nodes without incoming edges come first
    # (their order relative to one another isn't important)
    queue = deque([n for n in nodes if indegree[n] == 0])
    topo = []
    while queue:
        n = queue.popleft()  # when we pop a node, it's ready to go in order
        topo.append(n)
        for c in children[n]:  # as the parent was ordered, we update the children's counts
            indegree[c] -= 1
            if indegree[c] == 0:  # occasionally, we are done and the child can be queued 
                queue.append(c)

    if len(topo) != len(nodes):
        raise ValueError("Graph contains a cycle — topological sort not possible")
    return tuple(topo)


def compute_ancestors(nodes: list, parents: dict, topo: list):
    """Return for each node a set containing its ancestors"""
    # start with empty sets
    ancestors = {n: set() for n in nodes}
    # go in topological order, so we can build ancestor sets incrementally
    # we collect direct ancestors, that is, parents, and the ancestors of those parents
    # (topological order ensures we have that information ready to be used)
    for n in topo: 
        for p in parents[n]: # parents are ancestors
            ancestors[n].add(p) 
            # ancestors of parent are also ancestors of n            
            ancestors[n].update(ancestors[p])  

    return ancestors

def compute_descendants(nodes, children, topo):
    """Return for each node a set containing its descendants"""
    # start with empty sets
    descendants = {n: set() for n in nodes}
    # go in *reversed* topological order, so we can build descendant sets incrementally
    # we collect direct descendants, that is, children, and the descendants of those children
    # (reversed topological order ensures we have that information ready to be used)    
    for n in reversed(topo):
        for c in children[n]: # children are descendants
            descendants[n].add(c)
            # descendants of children are also descendants of n
            descendants[n].update(descendants[c])
    
    return descendants


class DAG:
    """
    A container for a directed acyclic graph.

    The container holds nodes that are string objects 
    (or, at least, that overwrite the __str__ method).

    We store a tuple of nodes and a tuple of edges.
    For convenience of various DAG algs, we also store adjacency maps (for parents and children), 
    as well as a topological order of the nodes. 
    """

    def __init__(self, nodes: list, edges: list):
        """
        nodes: a list of string objects (or objects supporting str(obj))
        edges: a list of edges, each represented as a (parent, child) pair
        """
        self.nodes = tuple(str(node) for node in nodes)
        self.edges = tuple((str(parent), str(child)) for parent, child in edges)
        self.children, self.parents = build_adjacency(self.nodes, self.edges)
        self.topo = topological_sort(self.nodes, self.children, self.parents)
        self.ancestors = compute_ancestors(self.nodes, self.parents, self.topo)
        self.descendants = compute_descendants(self.nodes, self.children, self.topo)

    def enumerate_trails(self, start, end, visited=None):
        """
        Return a generator for all trails between start and end.
        
        start: a node
        end: a node
        visited: a set of visited nodes (used in recursive calls)
        
        Note that enumerating 1 path is always tractable (scales with the size of the DAG), 
        but enumerating all paths between two nodes is in general intractable 
        (worst case is exponential in the number of nodes in the DAG). 
        Using this for some visualisations is fine, but for algorithms like d-sep and such
        this isn't the best way to do things. 
        """
        if visited is None:
            visited = set()
        visited.add(start)
        if start == end:
            yield [start]
        else:
            neighbors = self.parents[start] | self.children.get(start, set())
            for nb in neighbors:
                if nb not in visited:
                    for trail in self.enumerate_trails(nb, end, visited.copy()):
                        yield [start] + trail
    
    def __str__(self):
        """Generate a view of the graph using tabulate"""
        rows = []
        for node in self.topo:
            rows.append([", ".join(str(u) for u in sorted(self.parents[node])), str(node)])            
        return tabulate(rows, headers=['parents', 'child'], tablefmt='grid')

    def __repr__(self):
        return str(self)