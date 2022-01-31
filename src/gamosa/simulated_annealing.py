from matplotlib import pyplot as plt
from metrics_suite import MetricsSuite
import networkx as nx
import math
import random as rand


class SimulatedAnnealing():

    def __init__(self, graph,
                        metrics_list=None, weights=None, mcdat="weighted_sum", 
                        initial_config="random",
                        cooling_schedule="linear_a",
                        t_min=0,
                        t_max=100,
                        alpha=0.6,
                        next_step="random",
                        n_nodes_random_step=1,
                        n_iters=1000,
                        ):
        
        self.initial_configs = {"random":self._initial_random,
                                "load":self._initial_load,
                                "x_axis":self._initial_x_axis,
                                "y_axis":self._initial_y_axis,
                                "grid":self._initial_grid,
        }

        self.next_steps = {"random":self._step_random,
                            "swap":self._step_swap,
                            "random_bounded": self._step_random_bounded,
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
        bb = graph_loader.get_bounding_box(self.graph)
        self.initial_step_bound_radius = graph_loader._euclidean_distance((bb[0][0],bb[0][1]), (bb[1][0],bb[1][1])) / 2
        self.step_bound_radius = self.initial_step_bound_radius

        graph_loader.draw_graph(self.initial_config)

    def _initial_load(self, graph):
        """Use the coordinates of a loaded graph drawing as the initial positions. Existing drawings 
        should already have positions assigned to nodes."""

        for node, attributes in graph.nodes(data=True):
            assert 'x' in attributes, f"Error: No X coordinate for node: {node}"
            assert 'y' in attributes, f"Error: No Y coordinate for node: {node}"

        return graph

    def _initial_x_axis(self, graph, distance=60):
        """Position nodes along the x axis"""

        position = 0
        for node in graph.nodes:
            graph.nodes[node]["x"] = position
            graph.nodes[node]["y"] = 0
            position += distance

        return graph

    def _initial_y_axis(self, graph, distance=60):
        """Position nodes along the x axis"""

        position = 0
        for node in graph.nodes:
            graph.nodes[node]["x"] = 0
            graph.nodes[node]["y"] = position
            position += distance
        
        return graph


    def _initial_grid(self, graph, w_distance=60, h_distance=60, w=0, h=0):
        
        i = 0
        j = 0
        n = graph.number_of_nodes()

        if w < 0:
            raise ValueError(f"w must be non negative but is {w}")
        if h < 0:
            raise ValueError(f"h must be non negative but is {h}")

        if w == 0 and h == 0:
            w = math.ceil(math.sqrt(n))
            h = math.ceil(math.sqrt(n))

        if w * h < n:
            raise ValueError(f"grid not large enough with width {w} and height {h}, need at least {n} spaces, currently there are {w*h}")

        positions = []
        for i in range(h):
            for j in range(w):
                positions.append((j*w_distance, i*h_distance))

        #positions = positions[:n]

        for node, pos in zip(graph.nodes, positions):
            graph.nodes[node]["x"] = pos[0]
            graph.nodes[node]["y"] = pos[1]
                
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
            graph.nodes[random_node]["x"], graph.nodes[random_node]["y"]  = random_x, random_y

        return graph

    def _step_swap(self, graph):
        """Swap the position of two random nodes"""
        a, b = rand.sample(list(graph.nodes), 2)
        
        temp_x, temp_y = graph.nodes[a]['x'], graph.nodes[a]['y']
        graph.nodes[a]['x'], graph.nodes[a]['y'] = graph.nodes[b]['x'], graph.nodes[b]['y']
        graph.nodes[b]['x'], graph.nodes[b]['y'] = temp_x, temp_y

        return graph


    def _step_random_bounded(self, graph):
        """Move the position of n random nodes to a new random position, bounded by a circle of decreasing size, where n is defined 
        in self.n_nodes_random_step"""
        
        ms = MetricsSuite(graph)
        bb = ms.get_bounding_box()


        for random_node in rand.sample(list(graph.nodes), self.n_nodes_random_step):
            r = self.step_bound_radius * math.sqrt(rand.random())
            theta = rand.random() * 2 * math.pi
            x = graph.nodes[random_node]["x"] + r * math.cos(theta)
            y = graph.nodes[random_node]["y"] + r * math.sin(theta)
            graph.nodes[random_node]["x"], graph.nodes[random_node]["y"]  = x, y

        self.step_bound_radius -= self.initial_step_bound_radius / self.n_iters
        if self.step_bound_radius < 5:
            self.step_bound_radius = 5

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
        if i == 0:
            return self.t_max

        new_t = self.t_max / (self.alpha * math.log(i + 1))
        
        return new_t if new_t <= self.t_max else self.t_max

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
                #print(f"{i}: best")


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


    def plot_temperatures2(self):


        plt.figure()
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')

        
        pos_lin_a = []
        temp_lin_a = []
        
        pos_lin_m = []
        temp_lin_m = []

        pos_quad_a = []
        temp_quad_a = []

        pos_quad_m = []
        temp_quad_m = []

        pos_log = []
        temp_log = []

        pos_exp = []
        temp_exp = []


        t = self.t_max
        for n in range(self.n_iters):
            t = self.cooling_schedules["linear_a"](n)
            pos_lin_a.append(n)
            temp_lin_a.append(t)

        t = self.t_max
        for n in range(self.n_iters):
            t = self.cooling_schedules["linear_m"](n)
            pos_lin_m.append(n)
            temp_lin_m.append(t)


        t = self.t_max
        for n in range(self.n_iters):
            t = self.cooling_schedules["quadratic_a"](n)
            pos_quad_a.append(n)
            temp_quad_a.append(t)


        t = self.t_max
        for n in range(self.n_iters):
            t = self.cooling_schedules["quadratic_m"](n)
            pos_quad_m.append(n)
            temp_quad_m.append(t)


        t = self.t_max
        for n in range(self.n_iters):
            t = self.cooling_schedules["exponential"](n)
            pos_exp.append(n)
            temp_exp.append(t)


        t = self.t_max
        for n in range(self.n_iters):
            t = self.cooling_schedules["logarithmic"](n)
            pos_log.append(n)
            temp_log.append(t)


        plt.plot(pos_lin_a, temp_lin_a, label = "linear additive")
        plt.plot(pos_lin_m, temp_lin_m, label = "linear multiplicative")
        plt.plot(pos_quad_a, temp_quad_a, label = "quadratic additive")
        plt.plot(pos_quad_m, temp_quad_m, label = "quadratic multiplicative")
        plt.plot(pos_exp, temp_exp, label = "exponential")
        plt.plot(pos_log, temp_log, label = "logarithmic")



        plt.title("Cooling Schedules")
        plt.legend()

        plt.show()