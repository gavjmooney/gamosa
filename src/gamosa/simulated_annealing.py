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