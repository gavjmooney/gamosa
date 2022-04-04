from matplotlib import pyplot as plt
from metrics_suite import MetricsSuite
import networkx as nx
import math
import random as rand


class SimulatedAnnealing():

    def __init__(self, 
                    metric_suite,
                    initial_config="random",
                    cooling_schedule="linear_a",
                    t_min=0,
                    t_max=100,
                    alpha=0.6,
                    next_step="random",
                    n_nodes_random_step=1,
                    n_iters=1000,
                    initial_dist=60,
                    grid_w=0,
                    grid_h=0,
                    n_polygon_sides=3,
                    ):
        
        self.initial_configs = {"random":self._initial_random,
                                "load":self._initial_load,
                                "x_axis":self._initial_x_axis,
                                "y_axis":self._initial_y_axis,
                                "xy_pos":self._initial_xy_pos,
                                "xy_neg":self._initial_xy_neg,
                                "grid":self._initial_grid,
                                "polygon":self._initial_polygon,
        }

        self.next_steps = {"random":self._step_random,
                            "swap":self._step_swap,
                            "random_bounded": self._step_random_bounded,
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
        
        self.metric_suite = metric_suite

        # Initial temperature parameters & cooling schedule
        self.t = t_max
        self.t_max = t_max
        self.t_min = t_min
        self.n_iters = n_iters
        self.alpha = alpha

        self.cooling_schedule = self.cooling_schedules[cooling_schedule]

        # Setup additional parameters for initial configurations
        self.initial_dist = initial_dist
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.n_polygon_sides = n_polygon_sides

        self.initial_config = self.initial_configs[initial_config](self.metric_suite.graph)

        # Setup additional parameters for next step
        self.n_nodes_random_step = n_nodes_random_step
        bb = self.metric_suite.get_bounding_box(self.metric_suite.graph)
        self.initial_step_bound_radius = self.metric_suite._euclidean_distance((bb[0][0],bb[0][1]), (bb[1][0],bb[1][1])) / 5
        self.step_bound_radius = self.initial_step_bound_radius

        self.next_step = self.next_steps[next_step]

        #self.metric_suite.draw_graph(self.initial_config)


    def _initial_load(self, graph):
        """Use the coordinates of a loaded graph drawing as the initial positions. Existing drawings 
        should already have positions assigned to nodes."""
        for node, attributes in graph.nodes(data=True):
            assert 'x' in attributes, f"Error: No X coordinate for node: {node}"
            assert 'y' in attributes, f"Error: No Y coordinate for node: {node}"

        return graph


    def _initial_x_axis(self, graph):
        """Position nodes along the x axis"""
        position = 0
        for node in graph.nodes:
            graph.nodes[node]["x"] = position
            graph.nodes[node]["y"] = 0
            position += self.initial_dist

        return graph


    def _initial_y_axis(self, graph):
        """Position nodes along the x axis"""
        position = 0
        for node in graph.nodes:
            graph.nodes[node]["x"] = 0
            graph.nodes[node]["y"] = position
            position += self.initial_dist
        
        return graph


    def _initial_xy_pos(self, graph):
        """Position nodes alone the line y=x"""
        position = 0
        for node in graph.nodes:
            graph.nodes[node]["x"] = position
            graph.nodes[node]["y"] = position
            position += self.initial_dist
        
        return graph


    def _initial_xy_neg(self, graph):
        """Position nodes alone the line y=-x"""
        position = 0
        for node in graph.nodes:
            graph.nodes[node]["x"] = position
            graph.nodes[node]["y"] = -position
            position += self.initial_dist
        
        return graph


    def _initial_grid(self, graph):
        """Position nodes in an orthogonal grid"""
        i = 0
        j = 0
        n = graph.number_of_nodes()

        # Check grid size
        if self.grid_w < 0:
            raise ValueError(f"w must be non negative but is {self.grid_w}")
        if self.grid_h < 0:
            raise ValueError(f"h must be non negative but is {self.grid_h}")

        if self.grid_w == 0 and self.grid_h == 0:
            self.grid_w = math.ceil(math.sqrt(n))
            self.grid_h = math.ceil(math.sqrt(n))

        if self.grid_w * self.grid_h < n:
            raise ValueError(f"grid not large enough with width {self.grid_w} and height {self.grid_h}, need at least {n} spaces, currently there are {self.grid_w*self.grid_h}")

        # Create the positions on the grid
        positions = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                positions.append((j*self.initial_dist, -i*self.initial_dist))

        
        # Assign the nodes to the position on the grid
        for node, pos in zip(graph.nodes, positions):
            graph.nodes[node]["x"] = pos[0]
            graph.nodes[node]["y"] = pos[1]
                
        return graph


    def _initial_polygon(self, graph):
        """Position the nodes around the perimeter of a regualr polygon"""
        # Check polygon has at least 3 sides
        if type(self.n_polygon_sides) != int:
            raise TypeError(f"n_polygon_sides must be of type int, not {type(self.n_polygon_sides)}")

        if self.n_polygon_sides <= 0:
            raise ValueError(f"n_polygon_sides must be non-negative")

        if self.n_polygon_sides == 1:
            for node in graph:
                graph.nodes[node]["x"], graph.nodes[node]["y"] = 0, 0
            return graph
        elif self.n_polygon_sides == 2:
            return self._initial_x_axis(graph)


        sides = self.n_polygon_sides
        n = graph.number_of_nodes()
        length = self.initial_dist * n / sides # Scale size of graph with number of nodes

        # First get the positions of corners of the polygons
        polygon_corners = []
        x, y = 0, 0
        angle = 0
        for i in range(sides):
            polygon_corners.append((x,y))
            angle += 360/sides
            x = x + (length * math.cos(math.radians(angle)))
            y = y + (length * math.sin(math.radians(angle)))
        
        # If there are less or the same nodes as there are sides to the polygon, just assign the nodes to corners postions
        if n <= sides:
            i = 0
            for node in graph:
                graph.nodes[node]["x"], graph.nodes[node]["y"] = polygon_corners[i]
                i += 1

            return graph

        # Otherwise, position the nodes evenly around the perimeter of the polygon
        nodes_per_side = round(n / sides)

        perfect_fit = False
        if n % sides == 0:
            perfect_fit = True

        # Loop over each side
        node_count = 0
        positions = []
        for i in range(sides):
            if i == sides-1:
                i = -1
                if not perfect_fit:
                    nodes_per_side = n - node_count

                    k=0
                    while k < nodes_per_side:
                        x = polygon_corners[i][0] * (1-k/nodes_per_side) + polygon_corners[i+1][0] * k/nodes_per_side
                        y = polygon_corners[i][1] * (1-k/nodes_per_side) + polygon_corners[i+1][1] * k/nodes_per_side
                        positions.append((x,y))
                        node_count += 1
                        k+=1
                    break

            # Place the nodes for the current side
            for j in range(nodes_per_side):
                x = (polygon_corners[i][0] * (1-(j/nodes_per_side))) + (polygon_corners[i+1][0] * (j/nodes_per_side))
                y = (polygon_corners[i][1] * (1-(j/nodes_per_side))) + (polygon_corners[i+1][1] * (j/nodes_per_side))

                positions.append((x,y))
                node_count += 1
                if node_count == n:
                    break
                
            if node_count == n:
                    break

        # Check if the nodes do not reach the final corner of the polygon
        need_fixed = 0
        for pos in polygon_corners:
            if pos not in positions:
                need_fixed += 1

        # Ensure all corners of the polygons have nodes in them by swapping some along the perimeter
        while need_fixed > 0:
            i = len(positions) - 1
            while i > 0:
                if positions[i] not in polygon_corners:
                    break
                i -= 1

            positions[i] = polygon_corners[-need_fixed]
            need_fixed -= 1

        # Assign the positions to the node attributes
        i = 0
        for node in graph:
            graph.nodes[node]["x"], graph.nodes[node]["y"] = positions[i]
            i += 1

        return graph


    def _midpoint(self, a, b):
        """Return the midpoint of two lines"""
        mid_x = (a[0] + b[0]) / 2
        mid_y = (a[1] + b[1]) / 2
        return (mid_x, mid_y)


    def _distance(self, a, b):
        """Return the euclidean distance between two points"""
        return math.sqrt((a[0]-b[0])**2 + (a[1] - b[1])**2)


    def _is_between(self, a, b, c):
        """Return true if point c is between points a and b (in a straight line)"""
        return math.isclose(self._distance(a,c) + self._distance(c,b), self._distance(a,b))


    def _initial_random(self, graph):
        """Assign nodes to random positions as the initial layout for the algorithm."""
        pos = nx.random_layout(graph)
        for k,v in pos.items():
            pos[k] = {"x":v[0]*graph.number_of_nodes()*20, "y":v[1]*graph.number_of_nodes()*20}

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
        num_metrics = len(self.metric_suite.initial_weights)
        num_perfect_metrics = 0
        for metric in self.metric_suite.initial_weights:
            if ms.metrics[metric]["value"] >= satisfactory_level:
                num_perfect_metrics += 1

        return num_perfect_metrics == num_metrics


    def anneal(self, print_stats=False):
        # Get initial config and set best layout to the initial layout
        best_ms = MetricsSuite(self.initial_config, self.metric_suite.initial_weights, self.metric_suite.mcdat)
        best = self.initial_config.copy()
        best_eval = best_ms.combine_metrics()
        self.initial_eval = best_eval

        curr, curr_eval = best, best_eval # Set the current layout to the initial layout

        self.n_accepted, self.n_total = 0, 0 # For statistical purposes

        for i in range(self.n_iters):
            if print_stats:
                print(i)

            self.n_total += 1
            
            # Propose a new candidate layout
            candidate = best.copy()
            candidate = self.next_step(candidate) # Using the next_step parameter to create a new layout
            candidate_ms = MetricsSuite(candidate, self.metric_suite.initial_weights, self.metric_suite.mcdat)
            candidate_eval = candidate_ms.combine_metrics() # Evaluate this layout

            # Decide whether to accept a worse solution as the new current solution probabalistically accoridng to the current temperature
            diff = candidate_eval - curr_eval
            if rand.random() < math.exp(-diff/self.t):
                curr, curr_eval = candidate, candidate_eval
                self.n_accepted += 1
            
            # Check if there is a new best solution
            if candidate_eval > best_eval:
                best, best_eval, best_ms = candidate, candidate_eval, candidate_ms

            self.t = self.cooling_schedule(i) # Update the temperature according to the cooling schedule
        
        if print_stats:
            print(f"Acceptance rate: {self.n_accepted/self.n_total if self.n_total > 0 else 0}")
            print(f"Initial eval: {self.initial_eval}")
            print(f"Best eval after {self.n_total} iterations: {best_eval}")
        return best


    def plot_temperatures(self):
        """Plot the cooling schedules"""
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


    def plot_temperatures_2(self):
        """Plot cooling schedules, without using loop for more customization"""

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


if __name__ == "__main__":
    pass