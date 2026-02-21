from torch_geometric.data import Data
import torch

def build_career_graph(transitions: list[dict]) -> tuple[Data, dict]:
    """
    Build a graph from career transitions.

    Returns:
        Data: PyTorch Geometric graph
        dict: job title → index mapping
    """

    if not transitions:
        # Return empty graph
        return Data(), {}

    # Step 1: create mapping
    jobs = set()
    for t in transitions:
        jobs.add(t["from_job"])
        jobs.add(t["to_job"])

    job_to_idx = {job: idx for idx, job in enumerate(jobs)}

    # Step 2: build edges
    edges = []
    years = []

    for t in transitions:
        src = job_to_idx[t["from_job"]]
        dst = job_to_idx[t["to_job"]]

        # Skip self-loops (design choice)
        if src == dst:
            continue

        edges.append([src, dst])
        years.append([t["years"]])

    if not edges:
        return Data(), job_to_idx

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(years, dtype=torch.float)

    graph = Data(edge_index=edge_index, edge_attr=edge_attr)

    return graph, job_to_idx


def get_neighbors(data: Data, node_idx: int) -> list[int]:
    """
    Return all neighbors reachable in one hop.
    """

    if data.edge_index is None:
        return []

    src_nodes = data.edge_index[0]
    dst_nodes = data.edge_index[1]

    neighbors = dst_nodes[src_nodes == node_idx]

    return neighbors.tolist()