#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dag.py
#   Author: xyy15926
#   Created: 2026-04-23 19:37:30
#   Updated: 2026-04-27 19:10:35
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import Set, Dict, List, Optional, Iterator, Self, Any, Iterable
from collections.abc import Hashable
from collections import deque

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

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
        self.upstream: Set[Self] = set()
        self.downstream: Set[Self] = set()

    def __repr__(self):
        return f"Node({self.id_})"

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id_ == other.id_
        return False

    @property
    def in_degree(self) -> int:
        return len(self.upstream)

    @property
    def out_degree(self) -> int:
        return len(self.downstream)

    @property
    def predecessors(self) -> Set[Self]:
        return set(self.upstream)

    @property
    def successors(self) -> Set[Self]:
        return set(self.downstream)

    def __rshift__(self, other: Self) -> Optional[Self]:
        """Add other node as one of the downstream."""
        cls = self.__class__
        if isinstance(other, cls):
            self.set_downstream(other)
            return other
        # Return None if setting multi-nodes as downstream.
        else:
            self.set_downstream(*other)
            return None

    def __lshift__(self, other: Self) -> Optional[Self]:
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

    def set_upstream(self, *nodes: Self):
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

    def set_downstream(self, *nodes: Self):
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


# %%
class DirectedGraph:
    """Graph"""
    def __init__(self):
        self._nodes: Dict[NID, Node] = {}
        self._edge_count: int = 0

    def __repr__(self):
        return f"Graph(nodes={self.node_count}, edges={self.edge_count})"

    def __contains__(self, node_id: NID | Node):
        if isinstance(node_id, Node):
            node_id = node_id.id_
        return node_id in self._nodes

    @staticmethod
    def from_root(root: Node):
        """Borad first search to construct a graph from the root node."""
        g = DirectedGraph()
        queue = deque([root, ])

        while queue:
            cur_node = queue.popleft()
            if cur_node in g:
                continue
            g._nodes[cur_node.id_] = cur_node

            for succ in cur_node.downstream:
                if succ not in g:
                    queue.append(succ)
                # Count only on downstream part of one edge.
                g._edge_count += 1
            for pred in cur_node.upstream:
                if pred not in g:
                    queue.append(pred)

        return g

# ------------------------------------------------------------------------
#                                   Node
# ------------------------------------------------------------------------
    def add_node(self, node: Node):
        """Add node."""
        node_id = node.id_
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} exists.")
        self._nodes[node_id] = node

    def get_node(self, node_id: Hashable) -> Optional[Node]:
        """Get node with node id."""
        return self._nodes.get(node_id)

    def remove_node(self, node_id: Hashable) -> bool:
        """Remove node and related edges.

        Return False if node is not in the graph.
        """
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]
        # Remove all in-edges.
        for pred in list(node.upstream):
            self.remove_edge(pred._id, node_id)
        # Remove all out-edges.
        for succ in list(node.downstream):
            self.remove_edge(node_id, succ.id)

        del self._nodes[node_id]
        return True

    def has_node(self, node_id: NID) -> bool:
        """If node in graph."""
        return node_id in self._nodes

    @property
    def nodes(self) -> Dict[NID, Node]:
        """Get all nodes."""
        return dict(self._nodes)

    @property
    def node_count(self) -> int:
        """Node count."""
        return len(self._nodes)

# ------------------------------------------------------------------------
#                                   Edge
# ------------------------------------------------------------------------
    def add_edge(self, from_id: NID, to_id: NID) -> bool:
        """Add edge.

        Return False if edge exists.
        """
        if from_id not in self._nodes:
            raise ValueError(f"Source node {from_id} doesn't exist.")
        if to_id not in self._nodes:
            raise ValueError(f"Target node {to_id} doesn't exist.")
        if from_id == to_id:
            raise ValueError("Self-cycle is not allowed.")

        from_node = self._nodes[from_id]
        to_node = self._nodes[to_id]

        # Return False if edge exists.
        if to_node in from_node.downstream:
            return False

        from_node.downstream.add(to_node)
        to_node.upstream.add(from_node)
        self._edge_count += 1
        return True

    def remove_edge(self, from_id: NID, to_id: NID) -> bool:
        """Remove edge.

        Return False if node or edge doesn't exist.
        """
        # Return False if node doesn't exist.
        if from_id not in self._nodes or to_id not in self._nodes:
            return False

        from_node = self._nodes[from_id]
        to_node = self._nodes[to_id]

        # Return False if edge doesn't exist.
        if to_node not in from_node.downstream:
            return False

        from_node.downstream.remove(to_node)
        to_node.upstream.remove(from_node)
        self._edge_count -= 1
        return True

    def has_edge(self, from_id: NID, to_id: NID) -> bool:
        """If edge exists in the graph."""
        if from_id not in self._nodes or to_id not in self._nodes:
            return False
        return self._nodes[to_id] in self._nodes[from_id].downstream

    @property
    def edge_count(self) -> int:
        """Edge count."""
        return self._edge_count

    def get_edges(self) -> List[tuple]:
        """Get all edges represented with tuple."""
        edges = []
        for node in self._nodes.values():
            for succ in node.downstream:
                edges.append((node.id, succ.id))
        return edges

# ------------------------------------------------------------------------
#                                   DAG
# ------------------------------------------------------------------------
    def topological_sort(self) -> Optional[List[List[NID]]]:
        """If graph is a directed acyclic graph.

        Kahn algorithm:
        1. Remove nodes with in-degree = 0 sequencelly.
        2. Mark the nodes removed.
        3. Nodes in cycle won't be removed. So the count of visited node must
          be equal to the count of all nodes in a DAG.

        Return:
        --------------------------
        List of List of Node ID in topological sort.
        """
        # Empty graph.
        if self.node_count == 0:
            return True

        # Get in-degrees of each node.
        in_degree = {nid: node.in_degree for nid, node in self._nodes.items()}

        # Put node in queue with in-degree = 0.
        cur_level = []
        next_level = [nid for nid, deg in in_degree.items() if deg == 0]
        topo_levels = []
        node_count = 0

        while not (len(cur_level) == 0 and len(next_level) == 0):
            topo_levels.append(next_level)
            node_count += len(next_level)
            cur_level = next_level
            next_level = []
            for cur_nid in cur_level:
                # Decrease in-degree of all neighbors acting as remove the current
                # node with in-degree = 0.
                for succ in self._nodes[cur_nid].downstream:
                    in_degree[succ.id_] -= 1
                    if in_degree[succ.id_] == 0:
                        next_level.append(succ.id_)
            cur_level = []

        # Nodes in cycle won't be removed. So the count of visited node must
        # be equal to the count of all nodes in a DAG.
        # And None should be return for a cyclic graph.
        return topo_levels if node_count == self.node_count else None

    def is_dag(self) -> bool:
        """If graph is a directed acyclic graph."""
        return self.topological_sort() is not None

    def find_cycle(self) -> Optional[List[str]]:
        """Find cycle in the graph.

        Find one cycle in the graph:
        1. Mark all nodes WHITE, uncertain.
        2. Deep first traverse, and mark the node in path GREY.
        3. If any succeeding node marked GREY during DFS, cycle found.
        4. If all succeeding node has been travesed and no cycle found,
          current node can't be in any cycle and mark it BLACK.
        5. Any path meeting a BLACK node returns directly.
        """
        # WHITE: Node uncertain
        # GRAY: Node in current DFS path
        # BLACK: Node not in any cycle
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}

        def dfs(node_id: str, path: List[str]) -> Optional[List[str]]:
            # Trace the deep first search path.
            path.append(node_id)
            # Mark the node in path GRAY.
            color[node_id] = GRAY

            for succ in self._nodes[node_id].downstream:
                # Cycle Found:
                # If any successor of the current node is marked GREY,
                # the path must be a cycle.
                if color[succ.id_] == GRAY:
                    cycle_start = path.index(succ.id_)
                    # Construct cycle: [succ.id_, ..., succ.id_]
                    cycle = path[cycle_start:] + [succ.id_, ]
                    return cycle
                elif color[succ.id_] == WHITE:
                    result = dfs(succ.id_, path)
                    if result:
                        return result

            # Pop current node out and mark it BLACK, not in any cycle.
            path.pop()
            color[node_id] = BLACK
            return None

        for nid in self._nodes:
            if color[nid] == WHITE:
                cycle = dfs(nid, [])
                if cycle:
                    return cycle

        return None

# ------------------------------------------------------------------------
#                                   Traverse
# ------------------------------------------------------------------------
    def bfs(self, start_id: NID) -> List[NID]:
        """Borad first search."""
        if start_id not in self._nodes:
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

            for succ in self._nodes[current_id].downstream:
                if succ.id_ not in visited:
                    queue.append(succ.id_)

        return result

    def dfs(self, start_id: NID) -> List[NID]:
        """Deep first search."""
        if start_id not in self._nodes:
            return []

        visited = set()
        result = []

        def _dfs(node_id: NID):
            visited.add(node_id)
            result.append(node_id)
            for succ in self._nodes[node_id].downstream:
                if succ.id_ not in visited:
                    _dfs(succ.id_)

        _dfs(start_id)
        return result

# ------------------------------------------------------------------------
#                                   Visualize
# ------------------------------------------------------------------------
    def visualize(self) -> str:
        """DAG visualization."""
        lines = ["\n🕸️  DAG:\n", ]
        levels = self.topological_sort()
        for i, level in enumerate(levels):
            prefix = "    " * i
            for nid in level:
                node = self._nodes[nid]
                deps = [u.id_ for u in node.upstream]
                dep_str = f" ← {', '.join(deps)}" if deps else "(Entrance)"
                lines.append(f"{prefix}└─ [{node.id_}]{dep_str}")

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Render graph to mermaid."""
        lines = ["graph TD;"]
        for node in self._nodes.values():
            for down in node.downstream:
                lines.append(f"    {node.id_}({node.id_}) --> "
                             f"{down.id_}({down.id_});")
        return "\n".join(lines)

