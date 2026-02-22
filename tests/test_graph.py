from src.graph import build_career_graph, get_neighbors

def test_graph_construction():
    transitions = [
        {"from_job": "Software Engineer", "to_job": "Senior Engineer", "years": 2.5},
        {"from_job": "Software Engineer", "to_job": "Tech Lead", "years": 4.0},
    ]

    graph, mapping = build_career_graph(transitions)

    assert len(mapping) == 3
    neighbors = get_neighbors(graph, mapping["Software Engineer"])
    assert len(neighbors) == 2