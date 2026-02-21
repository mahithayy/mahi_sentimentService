from graph import build_career_graph, get_neighbors

transitions = [
    {"from_job": "Software Engineer", "to_job": "Senior Engineer", "years": 2.5},
    {"from_job": "Software Engineer", "to_job": "Tech Lead", "years": 4.0},
]

graph, mapping = build_career_graph(transitions)

print("Mapping:", mapping)
print("Edge index:", graph.edge_index)
print("Neighbors of Software Engineer:", get_neighbors(graph, mapping["Software Engineer"]))