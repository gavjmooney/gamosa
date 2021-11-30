from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np


def test0():
    ms = MetricsSuite("test.graphml")

    #nx.draw(ms.graph)
    #plt.show()
    print(ms.graph)
    print()

    print(ms.graph.nodes(data=True))

    G = nx.sedgewick_maze_graph()
    pos = nx.random_layout(G)
    nx.set_node_attributes(G, pos, "pos")

    print()

    print(G.nodes(data=True))

def test1():
    G = nx.sedgewick_maze_graph()
    pos = nx.random_layout(G)
    nx.set_node_attributes(G, pos, "pos")

    ms = MetricsSuite(G)

    ms.calculate_metric("edge_orthogonality")
    #ms.calculate_metric("edge_crossing")

    for k,v in ms.metrics.items():
        print(f"{k}: {v['value']}")

def test2():
    G = nx.sedgewick_maze_graph()

    sa = SimulatedAnnealing(G, metrics_list=["node_orthogonality"], cooling_schedule="linear_m", n_iters=1000)

    G2 = sa.anneal()

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:v["pos"] for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    nx.draw(G2, pos={k:v["pos"] for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)

    plt.show()

def test3():
    G = nx.sedgewick_maze_graph()
    pos = nx.random_layout(G)
    nx.set_node_attributes(G, pos, "pos")

    ms = MetricsSuite(G)

    ms.calculate_metric("edge_orthogonality")
    #ms.calculate_metric("edge_crossing")

    for k,v in ms.metrics.items():
        print(f"{k}: {v['value']}")

    ms.draw_graph()

    ########################
    G = nx.sedgewick_maze_graph()
    pos = nx.bipartite_layout(G, G.nodes)
    nx.set_node_attributes(G, pos, "pos")

    ms = MetricsSuite(G)

    ms.calculate_metric("edge_orthogonality")
    #ms.calculate_metric("edge_crossing")

    for k,v in ms.metrics.items():
        print(f"{k}: {v['value']}")

    ms.draw_graph()

def test4():
    G = nx.sedgewick_maze_graph()

    weights = {"edge_crossing":1, "edge_orthogonality":1}
    sa = SimulatedAnnealing(G, metrics_list=list(weights.keys()), weights=weights, cooling_schedule="linear_m", n_iters=1000)

    G2 = sa.anneal()

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    nx.draw(G2, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)

    plt.show()

def test5():
    G = nx.sedgewick_maze_graph()
    #pos = nx.random_layout(G)
    pos = nx.bipartite_layout(G, G.nodes)
    nx.set_node_attributes(G, pos, "pos")

    ms = MetricsSuite(G)

    print(G.nodes(data=True))

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:v["pos"] for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    

    ms.calculate_metric("node_orthogonality")

    print(G.nodes(data=True))
    #ms.calculate_metric("edge_crossing")

    nx.draw(G, pos={k:v["pos"] for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax3)

    plt.show()

    for k,v in ms.metrics.items():
        print(f"{k}: {v['value']}")


def test7():


    sa = SimulatedAnnealing("test.graphml", metrics_list=["edge_crossing", "edge_orthogonality"], cooling_schedule="linear_m", n_iters=1000)

    G2 = sa.anneal()

    ms = MetricsSuite(G2)

    ms.write_graph("new.graphml", G2)

    ms.draw_graph()


def test8():
    

    ms = MetricsSuite("test.graphml")
    ms.draw_graph()

    ms.write_graph("out.graphml")


def test9():
    sa = SimulatedAnnealing("test.graphml", initial_config="load")

    g = sa.anneal()

    ms = MetricsSuite(g)

    ms.draw_graph()


def test10():
    ms = MetricsSuite("test.graphml", metrics_list=["angular_resolution"])
    ms.calculate_metric("angular_resolution")

    ms.pretty_print_metrics()

def test11():
    sa = SimulatedAnnealing("ar2.graphml", initial_config="load", metrics_list=["edge_orthogonality", "edge_crossing", "angular_resolution"])
    #sa = SimulatedAnnealing("test.graphml", initial_config="load", metrics_list=["edge_crossing"])

    g = sa.anneal()

    ms = MetricsSuite(g, metrics_list=["edge_orthogonality", "edge_crossing", "angular_resolution"])
    

    #ms.calculate_metrics()
    #ms.pretty_print_metrics()

    ms.draw_graph()

    #ms.write_graph("wtf.graphml")


def test12():

    ms = MetricsSuite("test.graphml", metrics_list=["edge_crossing"])
    ms.draw_graph()

    sa = SimulatedAnnealing(ms.graph)
    ms.calculate_metrics()
    ms.pretty_print_metrics()

    print(ms.metrics["edge_crossing"]["num_crossings"])

    print(ms.graph.edges)
    print(ms.graph.edges())


def test13():
    ms = MetricsSuite("test.graphml", metrics_list=["edge_crossing"])
    ms.draw_graph()

    ms.crosses_promotion()

    ms.draw_graph(ms.graph_cross_promoted)
    ms.write_graph("newone.graphml", ms.graph_cross_promoted)

    ms.pretty_print_nodes(ms.graph_cross_promoted)

def test14():
    ms = MetricsSuite("test.graphml", metrics_list=["edge_crossing"])

    print(ms._find_bisectors(ms.graph))


def test15():
    ms = MetricsSuite("ar.graphml", metrics_list=["edge_crossing"])

    print(ms.symmetry())

def test16():
    G = nx.sedgewick_maze_graph()
    #ms = MetricsSuite(G)
    pos = nx.random_layout(G)
    for k,v in pos.items():
        pos[k] = {"x":v[0], "y":v[1]}

    nx.set_node_attributes(G, pos)

    #ms.draw_graph(G)
    ms = MetricsSuite("angres.graphml")
    #ms.calculate_metric("angular_resolution")
    #ms.pretty_print_metrics()
    G = ms.graph

    #weights = { "angular_resolution":1, "edge_length":1}   
    #weights = { "gabriel_ratio":1, "node_resolution":1}   
    #weights = {"edge_length":1, "edge_crossing":2, "node_resolution":1, "angular_resolution":1, "gabriel_ratio":1}
    weights = { "edge_crossing":1}
    #weights = { "node_resolution":1, "edge_length":1 }
    sa = SimulatedAnnealing("angres.graphml", metrics_list=list(weights.keys()), weights=weights, cooling_schedule="linear_m", n_iters=10000, initial_config="load")

    G2 = sa.anneal()
    

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    
    nx.draw(G2, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)

    ms2 = MetricsSuite(G2, metrics_list=weights.keys())
    ms2.calculate_metrics()
    ms2.pretty_print_metrics()
    #ms.draw_graph(G2)
    plt.show()


def test17():
    ms = MetricsSuite("..\\..\\graphs\\moon\\angres3.graphml", metrics_list=["edge_crossing"])
    #print(ms.get_bounding_box())
    #ms.node_area()
    #print(ms.graph.nodes)
    # a = ms.graph.nodes['n0']['x'], ms.graph.nodes['n0']['y']
    # b = ms.graph.nodes['n3']['x'], ms.graph.nodes['n3']['y']
    # print(ms._euclidean_distance(a, b))
    #print(ms.node_resolution())
    #print(ms.edge_length())
    # for edge in ms.graph.edges:
    #     print(ms._midpoint(edge[0], edge[1]))
    #print(ms.gabriel_ratio())
    #print(ms.symmetry())
    #print(ms._circles_intersect(2, 1, 4, 1, 2, 1))
    print(ms.angular_resolution(True))
    ms.draw_graph()
    #print(ms.crossing_angle())


def test18():
    ms = MetricsSuite("ar.graphml")
    #a, b, c = ms.graph.nodes
    #print(ms._are_collinear(a,b,c,ms.graph))

    # print(ms._rel_point_line_dist(-2, 4, -1, -1))
    # print(ms._rel_point_line_dist2(-2, 4, -1, -1))
    ms.calculate_metric("angular_resolution")
    ms.pretty_print_metrics()

