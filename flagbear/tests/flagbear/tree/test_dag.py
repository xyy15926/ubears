#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dag.py
#   Author: xyy15926
#   Created: 2026-04-23 22:19:32
#   Updated: 2026-04-30 08:46:30
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import pytest

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, force=True)
    from importlib import reload
    from flagbear.tree import dag
    reload(dag)

from itertools import chain
from flagbear.tree.dag import(
    Node, DirectedGraph,
    topological_sort,
    topological_sort_from_entry,
    find_cycle,
    visualize, to_mermaid
)


# %%
def test_Node_topo_cycle():
    node_a = Node("A")
    node_b = Node("B")
    node_c = Node("C")
    node_d = Node("D")
    node_e = Node("E")
    node_f = Node("F")

    node_a >> [node_b, node_c]
    node_d << [node_b, node_c]
    node_d >> node_e >> node_f
    node_b << node_f

    nodes = [node_a, node_b, node_c, node_d, node_e, node_f]
    nodes_2 = [node_a, node_c, node_b, node_d, node_e, node_f]
    nodes_nof = [node_a, node_c, node_b, node_d, node_e]
    nodes_nof_2 = [node_a, node_b, node_c, node_d, node_e]

    # Topo sort and find cycle on list of nodes.
    topo_sort = topological_sort(nodes)
    assert topo_sort is None
    cycle = find_cycle(nodes)
    assert set(cycle) == set([node_b, node_d, node_e, node_f])

    # Topo from `node_e`.
    topo_sort_e = topological_sort_from_entry(node_e)
    assert topo_sort_e is None

    # Remove edges.
    node_f.remove_downstream(node_b)
    topo_sort = list(chain(*topological_sort(nodes)))
    assert topo_sort == nodes or topo_sort == nodes_2

    # Topo only for specific node.
    topo_sort_e = list(chain(*topological_sort_from_entry(node_e)))
    assert topo_sort_e == nodes_nof or topo_sort_e == nodes_nof_2
    topo_sort_b = list(chain(*topological_sort_from_entry(node_b)))
    assert topo_sort_b == [node_a, node_b]
    topo_sort_c = list(chain(*topological_sort_from_entry(node_c)))
    assert topo_sort_c == [node_a, node_c]

    # Visualize.
    _vis = visualize(nodes)
    _mermaid = to_mermaid(nodes)


# %%
def test_directed_graph_and_node():
    g = DirectedGraph()
    for nid in list("ABCDE"):
        g.add_node(Node(nid))
    with pytest.raises(ValueError):
        g.add_node(Node("E"))
    assert g.node_count == 5
    # Add some edges and no cylce.
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    g.add_edge("D", "E")
    assert g.edge_count == 5
    # Can't add existing edge.
    assert not g.add_edge("D", "E")

    # Node info.
    node_b = g.get_node("B")
    assert node_b.in_degree == 1
    assert node_b.out_degree == 1
    assert [n.id_ for n in node_b.predecessors] == ["A",]
    assert [n.id_ for n in node_b.successors] == ["D", ]

    # DAG check on a DAG.
    topo_sort = list(chain(*g.topological_sort()))
    # Set is unordered so is the result of topological sort.
    assert topo_sort == list("ABCDE") or topo_sort == list("ACBDE")
    assert g.is_dag()
    cycle = g.find_cycle()
    assert cycle is None

    # BFS and DFS.
    bfs_ret = g.bfs("A")
    assert bfs_ret == list("ACBDE") or bfs_ret == list("ABCDE")
    dfs_ret = g.dfs("A")
    assert dfs_ret == list("ACDEB") or dfs_ret == list("ABDEC")

    # And a cycle.
    g.add_node(Node("F"))
    g.add_edge("E", "F")
    g.add_edge("F", "B")  # 形成环: B -> D -> E -> F -> B

    # DAG check on a cyclic graph.
    topo_sort = g.topological_sort()
    assert topo_sort is None
    assert not g.is_dag()
    cycle = g.find_cycle()
    assert set(cycle) == set("BDEF")

    # Remove edges.
    g.remove_edge("F", "B")
    assert g.is_dag()
    topo_sort = list(chain(*g.topological_sort()))
    assert topo_sort == list("ACBDEF") or topo_sort == list("ABCDEF")

    # Visualization
    _vis = g.visualize()
    _mermaid = g.to_mermaid()


# %%
def test_graph_from_nodes():
    node_a = Node("A")
    node_b = Node("B")
    node_c = Node("C")
    node_d = Node("D")
    node_e = Node("E")
    node_f = Node("F")

    node_a >> [node_b, node_c]
    node_d << [node_b, node_c]
    node_d >> node_e >> node_f
    node_b << node_f

    # Construct graph from any node.
    for node in [node_a, node_b, node_c, node_d, node_e, node_f]:
        g = DirectedGraph.from_root(node)
        assert g.edge_count == 7
        # Can't add existing edge.
        assert not g.add_edge("D", "E")

        # Node info.
        node_b = g.get_node("B")
        assert node_b.in_degree == 2
        assert node_b.out_degree == 1
        b_preds = [n.id_ for n in node_b.predecessors]
        assert b_preds == ["A", "F"] or b_preds == ["F", "A"]
        assert [n.id_ for n in node_b.successors] == ["D", ]

        # DAG check on a cyclic graph.
        topo_sort = g.topological_sort()
        assert topo_sort is None
        assert not g.is_dag()
        cycle = g.find_cycle()
        assert set(cycle) == set("BDEF")

    # Remove edges.
    g.remove_edge("F", "B")
    assert g.edge_count == 6
    assert g.is_dag()
    topo_sort = list(chain(*g.topological_sort()))
    assert topo_sort == list("ACBDEF") or topo_sort == list("ABCDEF")
