import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.operators.binary import difference
import numpy as np
import random as rand
import math
from write_graph import write_graphml_pos

class MetricsSuite():
    """A suite for calculating several metrics for graph drawing aesthetics, as well as methods for combining these into a single cost function.
    Takes as an argument an optional list of metrics to calculate (only edge crossings by default) and an optional method for combining them (weighted sum by default).
    Also takes an optional dictionary of metric:weight values defining the relative weight of each metric. Without this dictionary all weights are defaulted to 1"""

    def __init__(self, graph=None, metrics_list=None, weights=None, mcdat="weighted_sum"):

        self.metrics = {"edge_crossing": {"func":self.edge_crossing, "value":None, "weight":1},
                        "edge_orthogonality": {"func":self.edge_orthogonality, "value":None, "weight":1},
                        "node_orthogonality": {"func":self.node_orthogonality, "value":None, "weight":1},
                        "angular_resolution": {"func":self.angular_resolution, "value":None, "weight":1},
        
        } 
        self.mcdat_dict = {"weighted_sum":self._weighted_sum,
                           "weighted_prod":self._weighted_prod,
        
        }

        # Check all metrics given are valid
        if metrics_list is None:
            self.metrics_list = ["edge_crossing"]
        else:
            self.metrics_list = metrics_list
        
            for metric in metrics_list:
                assert metric in self.metrics, f"Unknown metric: {metric}. Available metrics: {list(self.metrics.keys())}"

        # Check metric combination strategy is valid
        assert mcdat in self.mcdat_dict, f"Unkown mcdat: {mcdat}. Available mcats: {list(self.mcdat_dict.keys())}"
        
        if graph is None:
            self.graph = self.load_graph_test()
        elif isinstance(graph, str):
            self.graph = self.load_graph(graph)
        else:
            self.graph = graph

        if weights:
            self.add_weights(weights)


        self.mcdat = mcdat


    def _weighted_prod(self):
        """Returns the weighted product of all metrics."""
        return math.prod(self.metrics[metric]["value"] * self.metrics[metric]["weight"] for metric in self.metrics_list)


    def _weighted_sum(self):
        """Returns the weighted sum of all metrics."""
        return sum(self.metrics[metric]["value"] * self.metrics[metric]["weight"] for metric in self.metrics_list) / len(self.metrics_list)
    

    def load_graph_test(self, nxg=nx.sedgewick_maze_graph):
        """Loads a test graph with a random layout."""
        #G = nx.sedgewick_maze_graph()
        G = nxg()
        pos = nx.random_layout(G)
        for k,v in pos.items():
            pos[k] = {"x":v[0], "y":v[1]}

        nx.set_node_attributes(G, pos)
        return G


    def load_graph(self, filename):
        """Loads a graph from a file."""
        G = nx.read_graphml(filename)
        G = G.to_undirected()

        for node in G.nodes:
            G.nodes[node]['x'] = float(G.nodes[node]['x'])
            G.nodes[node]['y'] = float(G.nodes[node]['y'])

        return G


    def write_graph_no_pos(self, filename, graph=None):
        """Writes a graph without preserving any information about node position."""
        if graph is None:
            graph = self.graph

        nx.write_graphml(graph, filename, named_key_ids=True)


    def write_graph(self, filename, graph=None, scale=True):
        """Writes a graph to GraphML format. Will not preserve all attributes of a graph loaded from GraphML."""
        if graph is None:
            graph = self.graph

        # If specified, scale the size of the graph to make it more suited to graphml format
        if scale:
            coords = []
            for node in graph:
                coords.append(abs(float((graph.nodes[node]['x']))))
                coords.append(abs(float((graph.nodes[node]['y']))))

            avg_dist_origin = sum(coords) / len(coords)
            
            # Note values are arbritrary
            if avg_dist_origin < 100:
                for node in graph:
                    graph.nodes[node]["x"] *= 750
                    graph.nodes[node]["y"] *= 750

        write_graphml_pos(graph, filename)


    def calculate_metric(self, metric):
        """Calculate the value of the given metric by calling the associated function."""
        self.metrics[metric]["value"] = self.metrics[metric]["func"]()


    def calculate_metrics(self):
        """Calculates the values of all metric defined in metrics_list."""
        for metric in self.metrics_list:
            self.calculate_metric(metric)


    def add_weights(self, weights):
        """Takes a dictionary of metric:weights and assigns that metric to its weight."""
        for metric, weight in weights.items():
            self._assign_weight(metric, weight)


    def _assign_weight(self, metric, weight):
        self.metrics[metric]["weight"] = weight


    def combine_metrics(self):
        """Combine several metrics based on the given multiple criteria descision analysis technique."""
        for metric in self.metrics_list:
            if self.metrics[metric]["value"] == None:
                # Possibly remove this and throw error instead, don't want to be calcualting metrics implicitly
                self.calculate_metric(metric)

        return self.mcdat_dict[self.mcdat]()


    def draw_graph(self, flip=True):
        """Draws the graph using standard networkx methods with matplotlib. Due to the nature of the coordinate systems used,
        Graphs will be flipped on the X axis. To see the graph the way it would be drawn in YeD, set flip to True"""
        if flip:
            pos={k:np.array((v["x"], 0-float(v["y"])),dtype=np.float32) for (k, v) in[u for u in self.graph.nodes(data=True)]}
        else:
            pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in self.graph.nodes(data=True)]}
        nx.draw(self.graph, pos=pos)
        plt.show()


    def pretty_print_metrics(self):
        for k,v in self.metrics.items():
            print(f"{k}: {v['value']}")

    def _on_opposite_sides(self, a, b, line):
        """Check if two lines pass the on opposite sides test. Return True if they do."""
        g = (line[1][0] - line[0][0]) * (a[1] - line[0][1]) - (line[1][1] - line[0][1]) * (a[0] - line[0][0])
        h = (line[1][0] - line[0][0]) * (b[1] - line[0][1]) - (line[1][1] - line[0][1]) * (b[0] - line[0][0])
        return g * h <= 0.0 and (a != line[1] and b != line[0] and a != line[0] and b != line[1])


    def _bounding_box(self, line_a, line_b):
        """Check if two lines pass the bounding box test. Return True if they do."""
        x1 = min(line_a[0][0], line_a[1][0])
        x2 = max(line_a[0][0], line_a[1][0])
        x3 = min(line_b[0][0], line_b[1][0])
        x4 = max(line_b[0][0], line_b[1][0])

        y1 = min(line_a[0][1], line_a[1][1])
        y2 = max(line_a[0][1], line_a[1][1])
        y3 = min(line_b[0][1], line_b[1][1])
        y4 = max(line_b[0][1], line_b[1][1])

        return x4 >= x1 and y4 >= y1 and x2 >= x3 and y2 >= y3


    def _intersect(self, line_a, line_b):
        """Check if two lines intersect by checking the on opposite sides and bounding box 
        tests. Return True if they do."""
        return (self._on_opposite_sides(line_a[0], line_a[1], line_b) and 
                self._on_opposite_sides(line_b[0], line_b[1], line_a) and 
                self._bounding_box(line_a, line_b))


    def edge_crossing(self):
        """Calculate the metric for the number of edge_crossing, scaled against the total
        number of possible crossings."""

        n = self.graph.number_of_nodes()
        m = self.graph.number_of_edges()
        c_all = (m * (m - 1))/2
        
        c_impossible = sum([(self.graph.degree[u] * (self.graph.degree[u] - 1)) for u in self.graph])/2
        
        c_mx = c_all - c_impossible
        
        covered = []
        c = 0
        for e in self.graph.edges:
            source = e[0]
            target = e[1]
            
            line_a_x1 = self.graph.nodes[source]["x"]
            line_a_y1 = self.graph.nodes[source]["y"]
            line_a_p1 = (line_a_x1, line_a_y1)
            
            line_a_x2 = self.graph.nodes[target]["x"]
            line_a_y2 = self.graph.nodes[target]["y"]
            line_a_p2 = (line_a_x2, line_a_y2)
            
            line_a = (line_a_p1, line_a_p2)
            
            for e2 in self.graph.edges:
                source = e2[0]
                target = e2[1]
                if e != e2:
                    line_b_x1 = self.graph.nodes[source]["x"]
                    line_b_y1 = self.graph.nodes[source]["y"]
                    line_b_p1 = (line_b_x1, line_b_y1)

                    line_b_x2 = self.graph.nodes[target]["x"]
                    line_b_y2 = self.graph.nodes[target]["y"]
                    line_b_p2 = (line_b_x2, line_b_y2)

                    line_b = (line_b_p1, line_b_p2)
                    
                    if self._intersect(line_a, line_b) and (line_a, line_b) not in covered:
                        covered.append((line_b, line_a))                  
                        c += 1

        return 1 - (c / c_mx) if c_mx > 0 else 1


    def edge_orthogonality(self):
        ortho_list = []

        for e in self.graph.edges:
            source = e[0]
            target = e[1]

            x1, y1 = self.graph.nodes[source]["x"], self.graph.nodes[source]["y"]
            x2, y2 = self.graph.nodes[target]["x"], self.graph.nodes[target]["y"]

            try:
                gradient = (y2 - y1) / (x2 - x1)
            except ZeroDivisionError:
                gradient = 0

            angle = math.degrees(math.atan(abs(gradient)))

            edge_ortho = min(angle, abs(90-angle), (180-angle)) /45
            ortho_list.append(edge_ortho)

        return 1 - (sum(ortho_list) / len(self.graph.edges))


    def angular_resolution(self):
        #print(self.graph.edges)
        min_list = []

        for node in self.graph.nodes:
            adj_nodes = list(self.graph.neighbors(node))

            temp = []
            


            ideal_angle = 360 / len(adj_nodes)

            for adj in adj_nodes:
                e1_source = node
                e1_target = adj

                #print((node,adj))

                e1_source_x, e1_source_y = float(self.graph.nodes[node]['x']), float(self.graph.nodes[node]['y'])
                e1_target_x, e1_target_y = float(self.graph.nodes[adj]['x']), float(self.graph.nodes[adj]['y'])

                initial_angle = math.degrees(math.atan2((e1_target_y - e1_source_y), (e1_target_x - e1_source_x)))

                for next_adj in adj_nodes:
                    if next_adj == adj:
                        continue

                    e2_source = node
                    e2_target = next_adj

                    e2_source_x, e2_source_y = float(self.graph.nodes[node]['x']), float(self.graph.nodes[node]['y'])
                    e2_target_x, e2_target_y = float(self.graph.nodes[next_adj]['x']), float(self.graph.nodes[next_adj]['y'])

                    new_angle = math.degrees(math.atan2((e2_target_y - e2_source_y), (e2_target_x - e2_source_x)))
                    diff = abs(initial_angle - new_angle)
                    if diff != 0:
                        temp.append(diff)

            if len(temp) == 0:
                continue
            minimum_angle = min(temp)
            scaled_minimum = abs((ideal_angle - minimum_angle) / ideal_angle)
            min_list.append(scaled_minimum)

        return 1 - (sum(min_list) / len(self.graph.nodes))


    # Doesn't work.
    def node_orthogonality(self):
        coord_set =[]

        # first_node = 0
        first_node = rand.sample(list(self.graph.nodes), 1)[0]
        
        min_x, min_y = self.graph.nodes[first_node]["x"], self.graph.nodes[first_node]["y"]

        for node in self.graph.nodes:
            x = self.graph.nodes[node]["x"]
            y = self.graph.nodes[node]["y"]
            
            if x < min_x:
                min_x = x
            elif y < min_y:
                min_y = y

        x_distance = abs(0 - float(min_x))
        y_distance = abs(0 - float(min_y))

        # Adjust graph so node with minimum coordinates is at 0,0
        for node in self.graph.nodes:
            self.graph.nodes[node]["x"] = float(self.graph.nodes[node]["x"]) - x_distance
            self.graph.nodes[node]["y"]= float(self.graph.nodes[node]["y"]) - y_distance


        # first_node = 0
        first_node = rand.sample(list(self.graph.nodes), 1)[0]
        
        min_x, min_y = self.graph.nodes[first_node]["x"], self.graph.nodes[first_node]["y"]
        max_x, max_y = self.graph.nodes[first_node]["x"], self.graph.nodes[first_node]["y"]

        for node in self.graph.nodes:
            x = self.graph.nodes[node]["x"]
            y = self.graph.nodes[node]["y"]

            coord_set.append(x)
            coord_set.append(y)

            gcd = int(float(coord_set[0]))
            for coord in coord_set[1:]:
                gcd = math.gcd(int(float(gcd)), int(float(coord)))

            if x > max_x:
                max_x = x
            elif x < min_x:
                min_x = x           

            if y > max_y:
                max_y = y            
            elif y < min_y:
                min_y = y 

        h = abs(max_y - min_y)
        w = abs(max_x - min_x)

        reduced_h = h / gcd
        reduced_w = w / gcd

        return len(self.graph.nodes) / ((reduced_w+1) * (reduced_h+1))



class SimulatedAnnealing():

    def __init__(self, graph,
                        metrics_list=None, weights=None, mcdat="weighted_sum", 
                        initial_config="random",
                        cooling_schedule="linear_a",
                        t_min=0,
                        t_max=100,
                        alpha=0.8,
                        next_step="random",
                        n_nodes_random_step=1,
                        n_iters=1000,
                        ):
        
        self.initial_configs = {"random":self._initial_random,
                                "load":self._initial_load,
                                #"x_align":self._inital_x_align,
        }

        self.next_steps = {"random":self._step_random,
                                #"load":self._step_random2,
                                #"x_align":self.something,
        }

        self.cooling_schedules = {"linear_a":self.additive_linear,
                                "linear_m":self.multiplicative_linear,
                                "quadratic_a":self.additive_quadratic,
                                "quadratic_m":self.multiplicative_quadratic,
                                "exponential":self.multiplicative_exponential,
                                "logarithmic":self.multiplicative_logarithmic,
        }

        # Check paramaters are valid
        assert initial_config in self.initial_configs, f"Unkown choice for initial_config: {initial_config}. Available choices: {list(self.initial_configs.keys())}"
        assert next_step in self.next_steps, f"Unkown choice for next_step: {next_step}. Available choices: {list(self.next_steps.keys())}"
        assert cooling_schedule in self.cooling_schedules, f"Unkown cooling_schedule: {cooling_schedule}. Available cooling schedules: {list(self.cooling_schedules.keys())}"
        
        graph_loader = MetricsSuite(graph)
        self.graph = graph_loader.graph
        if metrics_list is None:
            self.metrics_list = ["edge_crossing"]
        else:
            self.metrics_list = metrics_list
        self.weights = weights
        self.mcdat = mcdat

        self.initial_config = self.initial_configs[initial_config](self.graph)
        self.next_step = self.next_steps[next_step]
        self.cooling_schedule = self.cooling_schedules[cooling_schedule]
       
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.n_iters = n_iters
        self.alpha = alpha

        self.n_nodes_random_step = n_nodes_random_step

    def _initial_load(self, graph):
        """Use the coordinates of a loaded graph drawing as the initial positions. Existing drawings 
        should already have positions assigned to nodes."""

        for node, attributes in graph.nodes(data=True):
            assert 'x' in attributes, f"Error: No X coordinate for node: {node}"
            assert 'y' in attributes, f"Error: No Y coordinate for node: {node}"

        return graph
        

    def _initial_random(self, graph):
        """Assign nodes to random positions as the initial layout for the algorithm."""

        pos = nx.random_layout(graph)
        for k,v in pos.items():
            pos[k] = {"x":v[0], "y":v[1]}

        nx.set_node_attributes(graph, pos)

        return graph

    def _step_random(self, graph):
        """Move the position of n random nodes to a new random position, where n is defined 
        in self.n_nodes_random_step"""

        for random_node in rand.sample(list(graph.nodes), self.n_nodes_random_step):
            random_x = rand.uniform(0,1)
            random_y = rand.uniform(0,1)
            graph.nodes[random_node]["x"] = random_x
            graph.nodes[random_node]["y"] = random_y

        return graph


    def multiplicative_linear(self, i):
        return self.t_max / (1 + self.alpha * i)

    def additive_linear(self, i):
        return self.t_min + (self.t_max - self.t_min) * ((self.n_iters - i) / self.n_iters)

    def multiplicative_quadratic(self, i):
        return self.t_max / (1 + self.alpha * i**2)

    def additive_quadratic(self, i):
        return self.t_min + (self.t_max - self.t_min) * ((self.n_iters - i) / self.n_iters)**2

    def multiplicative_exponential(self, i):
        return self.t_max * self.alpha**i

    def multiplicative_logarithmic(self, i):
        return self.t_max / (self.alpha * math.log(i + 1))

    def _check_satisfactory(self, ms, satisfactory_level=1):
        """Check the condition of current metrics compared to a defined level."""
        # Possibly refactor to allow for different levels for each metric
        num_metrics = len(self.metrics_list)
        num_perfect_metrics = 0
        for metric in self.metrics_list:
            if ms.metrics[metric]["value"] >= satisfactory_level:
                num_perfect_metrics += 1

        return num_perfect_metrics == num_metrics



    def anneal(self):
        best_ms = MetricsSuite(self.initial_config, self.metrics_list, self.weights, self.mcdat)
        best = self.initial_config.copy()
        best_eval = best_ms.combine_metrics()

        curr, curr_eval = best, best_eval

        self.n_accepted, self.n_total = 0, 0 # for statistical purposes

        for i in range(self.n_iters):
            # If all the metrics are satisfactory, exit the algorithm early
            if self._check_satisfactory(best_ms):
                break

            self.n_total += 1
            
            candidate = best.copy()
            candidate = self.next_step(candidate)
            candidate_ms = MetricsSuite(candidate, self.metrics_list, self.weights, self.mcdat)
            candidate_eval = candidate_ms.combine_metrics()


            diff = candidate_eval - curr_eval

            if rand.random() < math.exp(-diff/self.t): # possibly make this yet another parameter?
                curr, curr_eval = candidate, candidate_eval
                self.n_accepted += 1


            if candidate_eval > best_eval:
                best, best_eval, best_ms = candidate, candidate_eval, candidate_ms

            self.t = self.cooling_schedule(i)
           
        print(f"Acceptance rate: {self.n_accepted/self.n_total if self.n_total < 0 else 0}")
        print(f"Best eval after {self.n_total} iterations: {best_eval}")
        return best

    def plot_temperatures(self):

        num_axis = len(self.cooling_schedules.keys())
        axis = [None for i in range(num_axis)]  

        num_rows = math.ceil(math.sqrt(num_axis))
        num_cols = math.ceil(num_axis/num_rows)

        plt.figure(constrained_layout=True)
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')

        for i, cs in enumerate(self.cooling_schedules):
            
            iterations = [j for j in range(self.n_iters)]
            temperatures = []
            t = self.t_max
            for n in iterations:
                t = self.cooling_schedules[cs](n+1)
                temperatures.append(t)

            

            plt.subplot(int(str(num_rows)+ str(num_cols) + str(i+1)))
            plt.plot(iterations, temperatures)
            plt.title(cs)
            
        plt.suptitle('Cooling Schedules')
        plt.show()



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

    ms = MetricsSuite("ar2.graphml", metrics_list=["node_orthogonality"])
    ms.calculate_metrics()
    ms.pretty_print_metrics()


if __name__ == "__main__":
    test12()

    # sa = SimulatedAnnealing(nx.sedgewick_maze_graph())
    # sa.plot_temperatures()

    #ms = MetricsSuite("test.graphml")
    #ms.draw_graph()



    