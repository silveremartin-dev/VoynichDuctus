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

            # Check 8 neighbors
            for dy, dx in self.NEIGHBORS_8:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx_ < W and skeleton[ny, nx_]:
                    # Add edge with euclidean length
                    dist = np.sqrt(dy * dy + dx * dx)
                    graph.add_edge((int(y), int(x)), (int(ny), int(nx_)), weight=float(dist))

        return graph

    def decompose_into_strokes(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """
        Decomposes graph into linear stroke segments between critical nodes (endpoints and junctions).
        """
        if graph.number_of_nodes() == 0:
            return []

        # Find critical nodes (degree != 2)
        critical_nodes = [n for n, deg in graph.degree() if deg != 2]

        strokes = []
        visited_edges = set()

        # Handle cycles/loops without critical nodes (e.g. clean closed loops 'o')
        if len(critical_nodes) == 0 and graph.number_of_nodes() > 0:
            # Pick arbitrary start node (top-left)
            start_node = min(graph.nodes(), key=lambda n: (n[0], n[1]))
            # Trace cycle
            cycle = list(nx.dfs_preorder_nodes(graph, source=start_node))
            if len(cycle) > 1:
                coords = [(n[0], n[1], graph.nodes[n].get("width", 1.0)) for n in cycle]
                strokes.append({
                    "stroke_id": "S000",
                    "type": "loop",
                    "points": coords,
                    "length": float(len(coords)),
                    "start": cycle[0],
                    "end": cycle[-1]
                })
            return strokes

        # Trace from every critical node along its incident edges
        stroke_counter = 0
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
