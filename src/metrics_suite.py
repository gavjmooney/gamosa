import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import random as rand
import math
from write_graph import write_graphml_pos


class MetricsSuite():
    """A suite for calculating several metrics for graph drawing aesthetics, as well as methods for combining these into a single cost function.
    Takes as an argument an optional list of metrics to calculate (only edge crossings by default) and an optional method for combining them (weighted sum by default).
    Also takes an optional dictionary of metric:weight values defining the relative weight of each metric. Without this dictionary all weights are defaulted to 1"""

    def __init__(self, graph=None, metrics_list=None, weights=None, mcdat="weighted_sum"):

        self.metrics = {"edge_crossing": {"func":self.edge_crossing, "value":None, "num_crossings":None, "weight":1},
                        "edge_orthogonality": {"func":self.edge_orthogonality, "value":None, "weight":1},
                        "node_orthogonality": {"func":self.node_orthogonality, "value":None, "weight":1},
                        "angular_resolution": {"func":self.angular_resolution, "value":None, "weight":1},
                        "symmetry": {"func":self.symmetry, "value":None, "weight":1},
                        "node_resolution": {"func":self.node_resolution, "value":None, "weight":1},
                        "edge_length": {"func":self.edge_length, "value":None, "weight":1},
        
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
        assert mcdat in self.mcdat_dict, f"Unknown mcdat: {mcdat}. Available mcats: {list(self.mcdat_dict.keys())}"
        
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
        total_weight = sum(self.metrics[metric]["weight"] for metric in self.metrics_list)
        return sum(self.metrics[metric]["value"] * self.metrics[metric]["weight"] for metric in self.metrics_list) / total_weight #len(self.metrics_list)
    

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


    def draw_graph(self, graph=None, flip=True):
        """Draws the graph using standard networkx methods with matplotlib. Due to the nature of the coordinate systems used,
        Graphs will be flipped on the X axis. To see the graph the way it would be drawn in YeD, set flip to True"""
        if graph is None:
            graph = self.graph

        if flip:
            pos={k:np.array((v["x"], 0-float(v["y"])),dtype=np.float32) for (k, v) in[u for u in graph.nodes(data=True)]}
        else:
            pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in graph.nodes(data=True)]}

        nx.draw(graph, pos=pos)
        plt.show()

    def pretty_print_metrics(self):
        for k,v in self.metrics.items():
            print(f"{k}: {v['value']}")

    def pretty_print_nodes(self, graph=None):
        if graph is None:
            graph = self.graph
        
        for n in graph.nodes(data=True):
            print(n)

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

        self.metrics["edge_crossing"]["num_crossings"] = c
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


    def _add_crossing_node(self, l1, l2, G, e, e2):

        x_diff = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])
        y_diff = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(x_diff, y_diff)

        if div == 0:
            return G

        d = (det(*l1), det(*l2))
        x = det(d, x_diff) / div
        y = det(d, y_diff) / div

        label = str(len(G.nodes()))

        G.add_node(label)

        G.nodes[label]["label"] = '\n'
        G.nodes[label]["shape_type"] = "ellipse"
        G.nodes[label]["x"] = x
        G.nodes[label]["y"] = y
        G.nodes[label]["type"] = "minor"

        G.add_edge(e[0], label)
        G.add_edge(label, e[1])

        G.add_edge(e2[0], label)
        G.add_edge(label, e2[1])

        G.remove_edge(e[0], e[1])
        G.remove_edge(e2[0], e2[1])

        return G

    def crosses_promotion(self):
        crosses_promoted_G = self.graph.copy()
        for node in crosses_promoted_G:
            crosses_promoted_G.nodes[node]["type"] = "major"

        if self.metrics["edge_crossing"]["num_crossings"] is None:
            self.calculate_metric("edge_crossing")

        num_crossings = self.metrics["edge_crossing"]["num_crossings"]

        second_covered = list()
        crossing_count = 0
        crossing_found = False

        while crossing_count != num_crossings:
            crossing_found = False


            edges = crosses_promoted_G.edges


            for e in edges:
                if e in second_covered:
                    continue
                    
                source, target = e[0], e[1]

                source_node = crosses_promoted_G.nodes[source]
                target_node = crosses_promoted_G.nodes[target]
                
                l1_p1_x = source_node["x"]
                l1_p1_y = source_node["y"]

                l1_p2_x = target_node["x"]
                l1_p2_y = target_node["y"]

                l1_p1 = (l1_p1_x, l1_p1_y)
                l1_p2 = (l1_p2_x, l1_p2_y)
                l1 = (l1_p1, l1_p2)

                for e2 in edges:
                    if e == e2:
                        continue

                    source2 = e2[0]
                    target2 = e2[1]

                    
                    source2_node = crosses_promoted_G.nodes[source2]
                    target2_node = crosses_promoted_G.nodes[target2]

                    l2_p1_x = source2_node["x"]
                    l2_p1_y = source2_node["y"]

                    l2_p2_x = target2_node["x"]
                    l2_p2_y = target2_node["y"]

                    l2_p1 = (l2_p1_x, l2_p1_y)
                    l2_p2 = (l2_p2_x, l2_p2_y)
                    l2 = (l2_p1, l2_p2)

                    if self._intersect(l1, l2) and (l1, l2) not in second_covered: #purpose of second covered?
                        crossing_count += 1
                        second_covered.append(e)
                        #second_covered.append((l1, l2))
                        #print(second_covered)
                        crosses_promoted_G = self._add_crossing_node(l1, l2, crosses_promoted_G, e, e2)
                        crossing_found = True
                        break

                if crossing_found:
                    break
        
        # What to do about saving crosses promoted graph? How will this work with other metrics?
        self.graph_cross_promoted = crosses_promoted_G
        return crosses_promoted_G
    
    def _find_bisectors(self, G):
        """Returns the set of perpendicular bisectors between every pair of nodes"""
        bisectors = []
        covered = []

        for n1 in G.nodes:
            n1_x = G.nodes[n1]["x"]
            n1_y = G.nodes[n1]["y"]

            for n2 in G.nodes:
                if n1 == n2 or (n1, n2) in covered:
                    continue

                n2_x = G.nodes[n2]["x"]
                n2_y = G.nodes[n2]["y"]

                midpoint_x = (n2_x + n1_x) / 2
                midpoint_y = (n2_y + n1_y) / 2

                try:
                    initial_gradient = (n2_y - n1_y) / (n2_x - n1_x)
                    perp_gradient = (1 / initial_gradient) * -1
                    c = midpoint_y - (perp_gradient * midpoint_x)

                except:
                    if (n2_x == n1_x):
                        perp_gradient = "x"
                        c = midpoint_y

                    elif (n2_y == n1_y):
                        perp_gradient = "y"
                        c = midpoint_x

                grad_c = (perp_gradient, c)

                bisectors.append(grad_c)
                covered.append((n2, n1))

        return set(bisectors)

    def _is_minor(self, node, G):
        return G.nodes[node]["type"] == "minor"

    def _point_line_dist(self, gradient, y_intercept, x, y):
        x = gradient * float(x)
        denom = math.sqrt(gradient**2 + 1)
        return (abs(x + float(y) + float(y_intercept))) / denom

    def _rel_point_line_dist(self, gradient, y_intercept, x ,y):
        gradient *= -1
        y_intercept *= -1

        x = gradient * float(x)
        denom = math.sqrt(gradient**2 + 1)
        return (x + float(y) + float(y_intercept)) / denom

    def _same_position(self, n1, n2, G):
        return (G.nodes[n1]["x"] == G.nodes[n2]["x"] and G.nodes[n1]["y"] == G.nodes[n2]["y"])

    def _is_positive(self, x):
        return x > 0

    def _mirror(self, axis, e1, e2, G):
        e1_p1_x, e1_p1_y = G.nodes[e1[0]]["x"], G.nodes[e1[0]]["y"]
        e1_p2_x, e1_p2_y = G.nodes[e1[1]]["x"], G.nodes[e1[1]]["y"]

        e2_p1_x, e2_p1_y = G.nodes[e2[0]]["x"], G.nodes[e2[0]]["y"]
        e2_p2_x, e2_p2_y = G.nodes[e2[1]]["x"], G.nodes[e2[1]]["y"]

        P, Q, X, Y = e1[0], e1[1], e2[0], e2[1]

        if axis[0] == "x":
            p = axis[1] - e1_p1_y
            q = axis[1] - e1_p2_y
            x = axis[1] - e2_p1_y
            y = axis[1] - e2_p2_y
        elif axis[0] == "y":
            p = axis[1] - e1_p1_x
            q = axis[1] - e1_p2_x
            x = axis[1] - e2_p1_x
            y = axis[1] - e2_p2_x
        else:
            p = self._rel_point_line_dist(axis[0], axis[1], e1_p1_x, e1_p1_y)
            q = self._rel_point_line_dist(axis[0], axis[1], e1_p2_x, e1_p2_y)
            x = self._rel_point_line_dist(axis[0], axis[1], e2_p1_x, e2_p1_y)
            y = self._rel_point_line_dist(axis[0], axis[1], e2_p2_x, e2_p2_y)

        if e1 == e2:
            # Same edge
            return 0
        elif p == 0 and q == 0:
            # Edge on axis
            return 0
        elif y == 0 and x == 0:
            # Edge on other axis
            return 0
        elif self._same_position(P, X, G) and (p == 0 and x == 0):
            if abs(q) == abs(y) and (self._is_positive(q) != self._is_positive(y)):
                # Shared node on axis but symmetric
                return 1
        elif self._same_position(P, Y, G) == True and (p == 0 and y == 0):
            if abs(q) == abs(x) and (self._is_positive(q) != self._is_positive(x)):
                # Shared node on axis but symmetric
                return 1
        elif self._same_position(Q, Y, G) == True and (q == 0 and y == 0):
            if abs(p) == abs(x) and (self._is_positive(x) != self._is_positive(p)):
                # Shared node on axis but symmetric
                return p
        elif self._same_position(Q, X, G) and (q == 0 and x == 0):
            if abs(p) == abs(y) and (self._is_positive(p) != self._is_positive(y)):
                # Shared node on axis but symmetric
                return 1
        elif self._is_positive(p) != self._is_positive(q):
            # Edge crosses axis
            return 0
        elif self._is_positive(x) != self._is_positive(y):
            # Other edge crosses axis
            return 0
        elif (abs(p) == abs(x) and abs(q) == abs(y)) and (self._is_positive(p) != self._is_positive(x)) and (self._is_positive(q) != self._is_positive(y)):
            # Distances are equal and signs are different
            return 1
        elif (abs(p) == abs(y) and abs(x) == abs(q)) and (self._is_positive(p) != self._is_positive(y)) and (self._is_positive(x) != self._is_positive(q)):
            # Distances are equal and signs are different
            return 1
        else:
            return 0

    def _sym_value(self, e1, e2, G):
            P = e1[0]
            Q = e1[1]
            X = e2[0]
            Y = e2[1]

            
            if self._is_minor(P, G) == self._is_minor(X, G) and self._is_minor(Q, G) == self._is_minor(Y, G):
                # P=X and Q=Y
                return 1
            elif self._is_minor(P, G) == self._is_minor(Y, G) and self._is_minor(X, G) == self._is_minor(Q, G):
                # P=Y and X=Q
                return 1
            elif self._is_minor(P, G) == self._is_minor(X, G) and self._is_minor(Q, G) != self._is_minor(Y, G):
                # P=X but Q != Y
                return 0.5
            elif self._is_minor(P, G) == self._is_minor(Y, G) and self._is_minor(Q, G) != self._is_minor(X, G):
                # P=Y but Q!=X
                return 0.5
            elif self._is_minor(P, G) != self._is_minor(X, G) and self._is_minor(Q, G) != self._is_minor(Y, G):
                # P!=X and Q!=Y
                return 0.25
            elif self._is_minor(P, G) != self._is_minor(Y, G) and self._is_minor(Q, G) != self._is_minor(X, G):
                # P!=Y and Q!=X
                return 0.25
            elif self._is_minor(P, G) != self._is_minor(X, G) and self._is_minor(Q, G) == self._is_minor(Y, G):
                return 0.5

    def _subgraph_to_points(self, subgraph, G):
        points = []

        for p in subgraph:
            for q in p:
                p1_x, p1_y = G.nodes[q[0]]["x"], G.nodes[q[0]]["y"]
                p2_x, p2_y = G.nodes[q[1]]["x"], G.nodes[q[1]]["y"]
            
            points.append((p1_x, p1_y))
            points.append((p2_x, p2_y))

        return points

    def _graph_to_points(self, G):
        points = []

        for n in G.nodes:
            p1_x, p1_y = G.nodes[n]["x"], G.nodes[n]["y"]
            points.append((p1_x, p1_y))

        return points


    def symmetry(self, G=None):
        if G is None:
            G = self.crosses_promotion()

        
        axes = self._find_bisectors(G)

        total_area = 0
        total_sym = 0

        for a in axes:
            covered = []
            num_mirror = 0
            sym_val = 0
            subgraph = []

            for e1 in G.edges:
                for e2 in G.edges:
                    if e1 == e2:
                        continue
                    
                    if self._mirror(a, e1, e2, G) == 1:
                        num_mirror += 1
                        sym_val += self._sym_value(e1, e2, G)
                        subgraph.append((e1, e2))

                if num_mirror >= 2 :
                    points = self._subgraph_to_points(subgraph, G)
                    if len(points) <= 2:
                        break

                    conv_hull = ConvexHull(points, qhull_options="QJ")
                    sub_area = conv_hull.volume
                    total_area += sub_area

                    total_sym += (sym_val * sub_area) / len(subgraph)

        whole_area_points = self._graph_to_points(G)

        whole_hull = ConvexHull(whole_area_points)
        whole_area = whole_hull.volume

        return total_sym / max(whole_area, total_area)


    def get_bounding_box(self, G=None):
        if G is None:
            G = self.graph

        first_node = rand.sample(list(G.nodes), 1)[0]
        min_x, min_y = G.nodes[first_node]["x"], G.nodes[first_node]["y"]
        max_x, max_y = G.nodes[first_node]["x"], G.nodes[first_node]["y"]

        for node in G.nodes:
            x = G.nodes[node]["x"]
            y = G.nodes[node]["y"]

            if x > max_x:
                max_x = x

            if x < min_x:
                min_x = x           

            if y > max_y:
                max_y = y        

            if y < min_y:
                min_y = y 

        return ((min_x, min_y), (max_x, max_y))

    # does nothing - implement something?
    def node_area(self):
        bb = self.get_bounding_box()
        area = (bb[1][0] - bb[0][0]) * (bb[1][1] - bb[0][1])

        ideal_node_distribution = area / len(self.graph.nodes)
        print(f"area: {area}")

    def _euclidean_distance(self, a, b):
        return math.sqrt(((b[0] - a[0])**2) + ((b[1] - a[1])**2))

    def node_resolution(self):

        first_node, second_node = rand.sample(list(self.graph.nodes), 2)
        a = self.graph.nodes[first_node]['x'], self.graph.nodes[first_node]['y']
        b = self.graph.nodes[second_node]['x'], self.graph.nodes[second_node]['y']

        min_dist = self._euclidean_distance(a, b)
        max_dist = min_dist
        for i in self.graph.nodes:
            for j in self.graph.nodes:
                if i == j:
                    continue
                
                a = self.graph.nodes[i]['x'], self.graph.nodes[i]['y']
                b = self.graph.nodes[j]['x'], self.graph.nodes[j]['y']

                d = self._euclidean_distance(a, b)
                #print(d)

                if d < min_dist:
                    min_dist = d

                if d > max_dist:
                    max_dist = d

        #r = 1 / math.sqrt(len(self.graph.nodes))
        #nr = min_dist / max_dist
        #return nr if nr > 0 else 0

        return min_dist / max_dist


    def edge_length(self):
        """Minimize the average deviation from ideal length, as in Ahmed et al."""

        ideal_edge_length = 0
        for edge in self.graph.edges:
            a = self.graph.nodes[edge[0]]['x'], self.graph.nodes[edge[0]]['y']
            b = self.graph.nodes[edge[1]]['x'], self.graph.nodes[edge[1]]['y']
            
            ideal_edge_length += self._euclidean_distance(a, b)

        
        # For unweighted graphs, set the ideal edge length to the average edge length
        ideal_edge_length = ideal_edge_length / self.graph.number_of_edges()

        
        edge_length_sum = 0

        for edge in self.graph.edges:
            a = self.graph.nodes[edge[0]]['x'], self.graph.nodes[edge[0]]['y']
            b = self.graph.nodes[edge[1]]['x'], self.graph.nodes[edge[1]]['y']
            edge_length_sum += ((self._euclidean_distance(a, b) - ideal_edge_length) / ideal_edge_length)**2
            #edge_length_sum += (abs(self._euclidean_distance(a, b) - ideal_edge_length) / ideal_edge_length)


        return -math.sqrt((edge_length_sum / self.graph.number_of_edges()))
        #return 1 - (edge_length_sum / self.graph.number_of_edges())
        #el = math.sqrt((edge_length_sum / self.graph.number_of_edges()))
        #return 1 - el if el < 1 else 0
        

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
        ms = MetricsSuite(graph)
        bb = ms.get_bounding_box()

        for random_node in rand.sample(list(graph.nodes), self.n_nodes_random_step):
            # random_x = rand.uniform(0,1)
            # random_y = rand.uniform(0,1)
            random_x = rand.uniform(bb[0][0],bb[1][0])
            random_y = rand.uniform(bb[0][1],bb[1][1])
            graph.nodes[random_node]["x"] = random_x
            graph.nodes[random_node]["y"] = random_y

        return graph

    def _step_swap(self, graph):
        """Swap the position of two random nodes"""
        a, b = rand.sample(list(graph.nodes), 2)
        
        temp_x, temp_y = graph.nodes[a]['x'], graph.nodes[a]['y']

        graph.nodes[a]['x'], graph.nodes[a]['y'] = graph.nodes[b]['x'], graph.nodes[b]['y']
        graph.nodes[b]['x'], graph.nodes[b]['y'] = graph.nodes[temp_x]['x'], graph.nodes[temp_y]['y']

        return graph

    def _step_random_bounded(self, graph, i):
        """Move the position of n random nodes to a new random position, bounded by a circle of decreasing size, where n is defined 
        in self.n_nodes_random_step"""
        pass


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
        self.initial_eval = best_eval

        curr, curr_eval = best, best_eval

        self.n_accepted, self.n_total = 0, 0 # for statistical purposes

        for i in range(self.n_iters):
            #print(i)
            # If all the metrics are satisfactory, exit the algorithm early
            #if self._check_satisfactory(best_ms):
            #    break

            self.n_total += 1
            
            candidate = best.copy()
            candidate = self.next_step(candidate)
            candidate_ms = MetricsSuite(candidate, self.metrics_list, self.weights, self.mcdat)
            candidate_eval = candidate_ms.combine_metrics()

            #candidate_ms.draw_graph()

            

            diff = candidate_eval - curr_eval
            #print(f"{rand.random()} < {math.exp(-diff/self.t)}")
            if rand.random() < math.exp(-diff/self.t): # possibly make this yet another parameter?
                curr, curr_eval = candidate, candidate_eval
                self.n_accepted += 1
                

            if candidate_eval > best_eval:
                best, best_eval, best_ms = candidate, candidate_eval, candidate_ms
                print(f"{i}: best")


            self.t = self.cooling_schedule(i)
           
        print(f"Acceptance rate: {self.n_accepted/self.n_total if self.n_total > 0 else 0}")
        print(f"Initial eval: {self.initial_eval}")
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
    ms = MetricsSuite("bb.graphml")
    G = ms.graph

    weights = { "edge_crossing":5, "node_resolution":1, "angular_resolution":1, "edge_length":1 }
    #weights = { "node_resolution":1, "edge_length":1 }
    sa = SimulatedAnnealing("bb.graphml", metrics_list=list(weights.keys()), weights=weights, cooling_schedule="linear_m", n_iters=1000, initial_config="load")

    G2 = sa.anneal()

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    
    nx.draw(G2, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)
    #ms.draw_graph(G2)
    plt.show()


def test17():
    ms = MetricsSuite("bb2.graphml", metrics_list=["edge_crossing"])
    #print(ms.get_bounding_box())
    #ms.node_area()
    #print(ms.graph.nodes)
    # a = ms.graph.nodes['n0']['x'], ms.graph.nodes['n0']['y']
    # b = ms.graph.nodes['n3']['x'], ms.graph.nodes['n3']['y']
    # print(ms._euclidean_distance(a, b))
    print(ms.node_resolution())
    print(ms.edge_length())

if __name__ == "__main__":
    test16()


    # sa = SimulatedAnnealing(nx.sedgewick_maze_graph())
    # sa.plot_temperatures()

    # ms = MetricsSuite("test.graphml")
    # print(ms._find_positive(4225))
    #ms.draw_graph()



    