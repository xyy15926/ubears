#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dag.py
#   Author: xyy15926
#   Created: 2026-04-23 19:37:30
#   Updated: 2026-05-16 21:22:55
#   Description:
# ---------------------------------------------------------

# %%
import logging
from collections import deque
from collections.abc import Iterable
from typing import Self

logger = logging.getLogger(__name__)

NID = str | int


# %%
class Node:
    """Graph Node.

    Attr:
    --------------------------------
    id_: Identifier of the node.
    upstream: Set of nodes that are directing to current node.
    downstream: Set of nodes that are directed by current node.
    """
    def __init__(self, node_id: NID):
        self.id_ = node_id
        self.upstream: set[Self] = set()
        self.downstream: set[Self] = set()

    def __repr__(self):
        """Return repr."""
        return f"Node({self.id_})"

    def __hash__(self):
        """Return hash."""
        return hash(self.id_)

    def __eq__(self, other):
        """Return whether equal."""
        if isinstance(other, Node):
            return self.id_ == other.id_
        return False

    @property
    def in_degree(self) -> int:
        """In-degree of the node."""
        return len(self.upstream)

    @property
    def out_degree(self) -> int:
        """Out-degree of the node."""
        return len(self.downstream)

    @property
    def predecessors(self) -> set[Self]:
        """Predecessors of the node."""
        return set(self.upstream)

    @property
    def successors(self) -> set[Self]:
        """Successors of the node."""
        return set(self.downstream)

    def __rshift__(self, other: Self) -> Self | None:
        """Add other node as one of the downstream."""
        cls = self.__class__
        if isinstance(other, cls):
            self.set_downstream(other)
            return other
        # Return None if setting multi-nodes as downstream.
        else:
            self.set_downstream(*other)
            return None

    def __lshift__(self, other: Self) -> Self | None:
        """Add other node as one of the upstream."""
        cls = self.__class__
        if isinstance(other, cls):
            self.set_upstream(other)
            return other
        # Return None if setting multi-nodes as upstream.
        else:
            self.set_upstream(*other)
            return None

    def __rrshift__(self, other: Iterable[Self]) -> Self:
        """Add other nodes in iterable as ones of the upstream.

        `__rrshift__` is defined so that `[Node, ] >> self` will work.
        """
        self << other
        return self

    def __rlshift__(self, other: Iterable[Self]) -> Self:
        """Add other nodes in iterable as ones of the downstream.

        `__rlshift__` is defined so that `[Node, ] << self` will work.
        """
        self >> other
        return self

    def set_upstream(self, *nodes: Self) -> Self:
        """Set upstream Nodes."""
        cls = self.__class__
        for node in nodes:
            if not isinstance(node, cls):
                raise ValueError(
                    f"Not {cls} error: {node} could not be set as the upstream."
                )
            self.upstream.add(node)
            node.downstream.add(self)
        return self

    def set_downstream(self, *nodes: Self) -> Self:
        """Set downstream nodes."""
        cls = self.__class__
        for node in nodes:
            if not isinstance(node, cls):
                raise ValueError(
                    f"No {cls} error: {node} could not be set as the upstream."
                )
            self.downstream.add(node)
            node.upstream.add(self)
        return self

    def remove_upstream(self, *nodes: Self) -> Self:
        """Remove upstream nodes."""
        for node in nodes:
            if node in self.upstream:
                self.upstream.remove(node)
                node.downstream.remove(self)
        return self

    def remove_downstream(self, *nodes: Self) -> Self:
        """Remove downstream nodes."""
        for node in nodes:
            if node in self.downstream:
                self.downstream.remove(node)
                node.upstream.remove(self)
        return self


# %%
class DirectedGraph:
    """Graph

    Attrs:
    -----------------------------
    nodes: Dict of node-id and node in graph.
    edge_count: Count of edges in graph.
    leaf_nodes: Nodes with no upstream nodes.
    """
    def __init__(self):
        self.nodes: dict[NID, Node] = {}
        self.edge_count: int = 0
        self.leaf_nodes: set[NID] = set()

    def __repr__(self) -> str:
        """Return repr."""
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"

    def __contains__(
        self,
        from_id: NID | Node,
        to_id: NID | None = None,
    ) -> bool:
        """Return whether node or edge exists."""
        if to_id is None:
            return self.has_node(from_id)
        else:
            return self.has_edge(from_id, to_id)

# ------------------------------------------------------------------------
#                                   Node NID
# ------------------------------------------------------------------------
    def add_node(self, node_id: NID):
        """Add node."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} exists.")
        self.nodes[node_id] = Node(node_id)
        self.leaf_nodes.add(node_id)

    def get_node(self, node_id: NID) -> Node | None:
        """Get node with node id."""
        return self.nodes.get(node_id)

    def remove_node(self, node_id: NID) -> bool:
        """Remove node and related edges.

        Return False if node is not in the graph.
        """
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        # Remove all in-edges.
        for pred in list(node.upstream):
            self.remove_edge(pred._id, node_id)
        # Remove all out-edges.
        for succ in list(node.downstream):
            self.remove_edge(node_id, succ.id_)

        del self.nodes[node_id]
        if node_id in self.leaf_nodes:
            self.leaf_nodes.remove(node_id)
        return True

    def has_node(self, node_id: NID | Node) -> bool:
        """If node in graph."""
        if isinstance(node_id, Node):
            node_id = node_id.id_
        return node_id in self.nodes

    @property
    def node_count(self) -> int:
        """Node count."""
        return len(self.nodes)

# ------------------------------------------------------------------------
#                                   Edge
# ------------------------------------------------------------------------
    def add_edge(self, from_id: NID, to_id: NID) -> bool:
        """Add edge.

        Return False if edge exists.
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node {from_id} doesn't exist.")
        if to_id not in self.nodes:
            raise ValueError(f"Target node {to_id} doesn't exist.")
        if from_id == to_id:
            raise ValueError("Self-cycle is not allowed.")

        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]

        # Return False if edge exists.
        if to_node in from_node.downstream:
            return False

        from_node.downstream.add(to_node)
        to_node.upstream.add(from_node)
        self.edge_count += 1
        if to_id in self.leaf_nodes:
            self.leaf_nodes.remove(to_id)
        return True

    def remove_edge(self, from_id: NID, to_id: NID) -> bool:
        """Remove edge.

        Return False if node or edge doesn't exist.
        """
        # Return False if node doesn't exist.
        if from_id not in self.nodes or to_id not in self.nodes:
            return False

        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]

        # Return False if edge doesn't exist.
        if to_node not in from_node.downstream:
            return False

        from_node.downstream.remove(to_node)
        to_node.upstream.remove(from_node)
        self.edge_count -= 1
        if len(to_node.upstream) == 0:
            self.leaf_nodes.add(to_id)
        return True

    def has_edge(self, from_id: NID, to_id: NID) -> bool:
        """If edge exists in the graph."""
        if from_id not in self.nodes or to_id not in self.nodes:
            return False
        return self.nodes[to_id] in self.nodes[from_id].downstream

    def get_edges(self) -> list[tuple]:
        """Get all edges represented with tuple."""
        edges = []
        for node in self.nodes.values():
            for succ in node.downstream:
                edges.append((node.id, succ.id))
        return edges

# ------------------------------------------------------------------------
#                                   Existed Nodes
# ------------------------------------------------------------------------
    def extend_with_nodes(self, *nodes: Node):
        """Extend graph with existing nodes.

        Borad first search to add all related nodes to a graph.
        """
        queue = deque(nodes)

        while queue:
            cur_node = queue.popleft()
            if cur_node in self:
                continue
            self.nodes[cur_node.id_] = cur_node
            if len(cur_node.upstream) == 0:
                self.leaf_nodes.add(cur_node.id_)

            for succ in cur_node.downstream:
                if succ not in self:
                    queue.append(succ)
                # Count only on downstream part of one edge.
                self.edge_count += 1
            for pred in cur_node.upstream:
                if pred not in self:
                    queue.append(pred)

    @staticmethod
    def from_nodes(*nodes: Node):
        """Borad first search to construct a graph from the root node."""
        g = DirectedGraph()
        g.extend_with_nodes(*nodes)
        return g

# ------------------------------------------------------------------------
#                                   DAG
# ------------------------------------------------------------------------
    def topological_sort(
        self,
        *entry: NID,
    ) -> list[list[NID]] | None:
        """Sort nodes in graph topologically.

        Params:
        ---------------------------
        entry: Entry nodes as the destination of the topo-sort.
        """
        if len(entry) == 0:
            toponodes = topological_sort(self.nodes.values())
        else:
            toponodes = topological_sort_from_entry(*entry)
        if toponodes is None:
            return None
        topo_nids = [[node.id_ for node in level] for level in toponodes]
        return topo_nids

    def is_dag(self) -> bool:
        """If graph is a directed acyclic graph."""
        return self.topological_sort() is not None

    def find_cycle(self) -> list[NID] | None:
        """Find cycle in the graph."""
        cyclenodes = find_cycle(self.nodes.values())
        if cyclenodes is None:
            return None
        cycle_nids = [node.id_ for node in cyclenodes]
        return cycle_nids

# ------------------------------------------------------------------------
#                                   Traverse
# ------------------------------------------------------------------------
    def bfs(self, start_id: NID) -> list[NID]:
        """Borad first search."""
        if start_id not in self.nodes:
            return []

        visited = set()
        queue = deque([start_id])
        result = []

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            result.append(current_id)

            for succ in self.nodes[current_id].downstream:
                if succ.id_ not in visited:
                    queue.append(succ.id_)

        return result

    def dfs(self, start_id: NID) -> list[NID]:
        """Deep first search."""
        if start_id not in self.nodes:
            return []

        visited = set()
        result = []

        def _dfs(node_id: NID):
            visited.add(node_id)
            result.append(node_id)
            for succ in self.nodes[node_id].downstream:
                if succ.id_ not in visited:
                    _dfs(succ.id_)

        _dfs(start_id)
        return result

# ------------------------------------------------------------------------
#                                   Visualize
# ------------------------------------------------------------------------
    def visualize(self) -> str:
        """DirectedGraph visualization."""
        return visualize(self.nodes.values())

    def to_mermaid(self) -> str:
        """Render nodes and edges to mermaid."""
        return to_mermaid(self.nodes.values())


# %%----------------------------------------------------------------------
#                                   Visualize
# ------------------------------------------------------------------------
def visualize(nodes: list[Node]) -> str:
    """Nodes and edges visualization."""
    lines = ["\n🕸️  DAG:\n", ]
    levels = topological_sort(nodes)
    if levels is None:
        raise RuntimeError("Cyclic graph can't be visualize.")
    for i, level in enumerate(levels):
        prefix = "    " * i
        for node in level:
            deps = [u.id_ for u in node.upstream]
            dep_str = f" ← {', '.join(deps)}" if deps else "(Entrance)"
            lines.append(f"{prefix}└─ [{node.id_}]{dep_str}")

    return "\n".join(lines)


def to_mermaid(nodes: list[Node]) -> str:
    """Render nodes and edges to mermaid."""
    lines = ["graph TD;"]
    for node in nodes:
        for down in node.downstream:
            lines.append(f"    {node.id_}({node.id_}) --> "
                         f"{down.id_}({down.id_});")
    return "\n".join(lines)


# %%----------------------------------------------------------------------
#                                   DAG
# ------------------------------------------------------------------------
def topological_sort(
    nodes: list[Node] | dict[Node, int],
    # dest: Node | NID = None,
) -> list[list[Node]] | None:
    """Sort a list of nodes topologically.

    Kahn algorithm:
    1. Remove nodes with in-degree = 0 sequencelly.
    2. Mark the nodes removed.
    3. Nodes in cycle won't be removed. So the count of visited node must
      be equal to the count of all nodes in a DAG.

    Params:
    --------------------------
    nodes: A list of nodes or a dict of nodes and theirs in-degrees.
    dest: Node or node-id to stop sort early.
      Namely, stop sort when the encountered the specified nodes.

    Return:
    --------------------------
    List of list of Node ID in topological sort.
    """
    # Get in-degrees of each node.
    if isinstance(nodes, dict):
        in_degree = nodes
    else:
        in_degree = {node: node.in_degree for node in nodes}

    # if dest is not None and isinstance(dest, Node):
    #     dest = dest.id_

    # Put node in queue with in-degree = 0.
    cur_level = []
    next_level = [node for node, deg in in_degree.items() if deg == 0]
    topo_levels = []
    node_count = 0

    while not (len(cur_level) == 0 and len(next_level) == 0):
        topo_levels.append(next_level)
        node_count += len(next_level)
        cur_level = next_level
        next_level = []
        for cur_node in cur_level:
            # Decrease in-degree of all neighbors acting as remove the current
            # node with in-degree = 0.
            for succ in cur_node.downstream:
                # `in_degree` may only contains only part of nodes, namely
                # subgraph, in a graph.
                if succ not in in_degree:
                    continue
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    # Stop early when encountering destination node.
                    # if dest is not None and succ.id_ == dest:
                    #     topo_levels.append([succ, ])
                    #     return topo_levels
                    next_level.append(succ)
        cur_level = []

    # Nodes in cycle won't be removed. So the count of visited node must
    # be equal to the count of all nodes in a DAG.
    # And None should be return for a cyclic graph.
    return topo_levels if node_count == len(nodes) else None


def topological_sort_from_entry(
    *entry: Node,
) -> list[list[Node]] | None:
    """Sort nodes linking to entry node topologically."""

    def bfs_backward(entry: Node) -> dict[Node: int]:
        """Borad first search to collect nodes linking to entry node."""
        queue = deque(entry)
        visited = {}

        while queue:
            cur_node = queue.popleft()
            if cur_node in visited:
                continue
            visited[cur_node] = cur_node.in_degree
            for pred in cur_node.upstream:
                if pred not in visited:
                    queue.append(pred)

        return visited

    in_degree = bfs_backward(entry)
    return topological_sort(in_degree)


# %%
def find_cycle(
    nodes: list[Node] | dict[NID, Node],
) -> list[str] | None:
    """Find cycle in the graph.

    Find one cycle in the graph:
    1. Mark all nodes WHITE, uncertain.
    2. Deep first traverse, and mark the node in path GREY.
    3. If any succeeding node marked GREY during DFS, cycle found.
    4. If all succeeding node has been traversed and no cycle found,
      current node can't be in any cycle and mark it BLACK.
    5. Any path meeting a BLACK node returns directly.
    """
    # WHITE: Node uncertain
    # GRAY: Node in current DFS path
    # BLACK: Node not in any cycle
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)

    def dfs(node: Node, path: list[Node]) -> list[Node] | None:
        # Trace the deep first search path.
        path.append(node)
        # Mark the node in path GRAY.
        color[node] = GRAY

        for succ in node.downstream:
            # Cycle Found:
            # If any successor of the current node is marked GREY,
            # the path must be a cycle.
            if color[succ] == GRAY:
                cycle_start = path.index(succ)
                # Construct cycle: [succ.id_, ..., succ.id_]
                cycle = path[cycle_start:] + [succ, ]
                return cycle
            elif color[succ] == WHITE:
                result = dfs(succ, path)
                if result:
                    return result

        # Pop current node out and mark it BLACK, not in any cycle.
        path.pop()
        color[node] = BLACK
        return None

    for node in nodes:
        if color[node] == WHITE:
            cycle = dfs(node, [])
            if cycle:
                return cycle

    return None
