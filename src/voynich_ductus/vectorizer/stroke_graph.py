"""
Converts skeleton pixel maps into topological graphs of nodes (endpoints, junctions) and stroke segments.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import networkx as nx


class StrokeGraphExtractor:
    """
    Builds a topological graph from 1-pixel skeletons where:
    - Degree 1 nodes: Endpoints (pen down / pen up candidates)
    - Degree 2 nodes: Path internal continuation pixels
    - Degree >= 3 nodes: Branching junctions and crossing intersections
    """

    NEIGHBORS_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def build_pixel_graph(self, skeleton: np.ndarray, stroke_widths: np.ndarray) -> nx.Graph:
        """
        Builds raw 8-connectivity pixel graph.
        """
        graph = nx.Graph()
        ys, xs = np.where(skeleton)
        H, W = skeleton.shape

        for y, x in zip(ys, xs):
            w = float(stroke_widths[y, x])
            graph.add_node((int(y), int(x)), y=int(y), x=int(x), width=w)

            # Check 8 neighbors, skipping redundant diagonal shortcut inside a 2x2 L-corner
            # so 8-connected thin curves do not spawn spurious degree-3 triangle junctions.
            for dy, dx in self.NEIGHBORS_8:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and skeleton[ny, nx_]:
                    if abs(dy) == 1 and abs(dx) == 1:
                        if skeleton[y + dy, x] or skeleton[y, x + dx]:
                            continue
                    dist = np.sqrt(dy * dy + dx * dx)
                    graph.add_edge((int(y), int(x)), (int(ny), int(nx_)), weight=float(dist))

        return graph

    @staticmethod
    def _trace_ordered_cycle(sub_g: nx.Graph) -> List[Tuple[int, int]]:
        """Traces a connected 2-regular cycle sequentially and closes it back at start."""
        nodes = list(sub_g.nodes())
        if not nodes:
            return []
        start_node = min(nodes, key=lambda n: (n[0] * 1.5 + n[1] * 2.0))
        cycle = [start_node]
        visited = {start_node}
        curr = start_node
        while True:
            nbs = [n for n in sub_g.neighbors(curr) if n not in visited]
            if not nbs:
                break
            nxt = nbs[0]
            visited.add(nxt)
            cycle.append(nxt)
            curr = nxt
        if len(cycle) >= 3 and start_node in sub_g.neighbors(cycle[-1]):
            cycle.append(start_node)
        return cycle

    def decompose_into_strokes(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """
        Decomposes graph into linear stroke segments between critical nodes (endpoints and junctions),
        preserving closed topological cycles (e.g. bowls/ovals like 'o') as sequential closed loops.
        """
        if graph.number_of_nodes() == 0:
            return []

        strokes = []
        visited_edges = set()
        stroke_counter = 0

        # 1. Extract pure degree-2 connected components first (even if another component has junctions!)
        for comp_nodes in nx.connected_components(graph):
            sub_g = graph.subgraph(comp_nodes)
            if all(deg == 2 for _, deg in sub_g.degree()):
                cycle = self._trace_ordered_cycle(sub_g)
                if len(cycle) > 1:
                    for k in range(len(cycle) - 1):
                        visited_edges.add(tuple(sorted([cycle[k], cycle[k + 1]])))
                    coords = [(n[0], n[1], graph.nodes[n].get("width", 1.0)) for n in cycle]
                    strokes.append({
                        "stroke_id": f"S{stroke_counter:03d}",
                        "type": "loop",
                        "points": coords,
                        "length": float(len(coords)),
                        "start": cycle[0],
                        "end": cycle[-1]
                    })
                    stroke_counter += 1

        # 2. Find critical nodes (degree != 2) for branched/open paths
        critical_nodes = [n for n, deg in graph.degree() if deg != 2]
        if not critical_nodes:
            return strokes

        for crit in critical_nodes:
            for neighbor in graph.neighbors(crit):
                edge_key = tuple(sorted([crit, neighbor]))
                if edge_key in visited_edges:
                    continue

                # Walk along degree-2 path until next critical node or dead end
                path = [crit, neighbor]
                visited_edges.add(edge_key)
                curr = neighbor
                prev = crit

                while graph.degree(curr) == 2:
                    next_nodes = [n for n in graph.neighbors(curr) if n != prev]
                    if not next_nodes:
                        break
                    nxt = next_nodes[0]
                    edge_k = tuple(sorted([curr, nxt]))
                    visited_edges.add(edge_k)
                    path.append(nxt)
                    prev = curr
                    curr = nxt
                    if curr in critical_nodes:
                        break

                coords = [(n[0], n[1], graph.nodes[n].get("width", 1.0)) for n in path]
                strokes.append({
                    "stroke_id": f"S{stroke_counter:03d}",
                    "type": "segment",
                    "points": coords,
                    "length": float(len(coords)),
                    "start": path[0],
                    "end": path[-1]
                })
                stroke_counter += 1

        return strokes
