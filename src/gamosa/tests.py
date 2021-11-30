from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np
from os.path import exists

import unittest

class TestMetricsSuite(unittest.TestCase):

    PATH = "..\\..\\graphs\\moon\\"

    def test_load(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename)
        self.assertEqual(6, ms.graph.number_of_nodes())
        self.assertEqual(5, ms.graph.number_of_edges())
        node_positions = ((v["x"], v["y"]) for k,v in ms.graph.nodes(data=True))
        known_positions = [(0.0, 0.0), (135.0, 0.0), (0.0, 135.0), (0.0, 270.0), (-135.0, 0.0), (0.0, -135.0)]
        
        for pos in node_positions:
            self.assertIn(pos, known_positions)

    def test_write(self):
        G = nx.sedgewick_maze_graph()
        
        for node, i in enumerate(G.nodes):
            G.nodes[node]["x"] = i
            G.nodes[node]["y"] = i

        ms = MetricsSuite(G)
        filename = self.PATH + "write_test.graphml"
        ms.write_graph(filename, scale=False)
        self.assertTrue(exists(filename))

        ms = MetricsSuite(filename)
        self.assertEqual(8, ms.graph.number_of_nodes())
        self.assertEqual(10, ms.graph.number_of_edges())
        node_positions = ((v["x"], v["y"]) for k,v in ms.graph.nodes(data=True))
        known_positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
        
        for pos in node_positions:
            self.assertIn(pos, known_positions)


    def test_ar1(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["angular_resolution"])
        ms.calculate_metric("angular_resolution")

        self.assertEqual(1, ms.metrics["angular_resolution"]["value"])

    def test_ar075(self):
        filename = self.PATH + "test_6_5_AR075.graphml"
        ms = MetricsSuite(filename, metrics_list=["angular_resolution"])
        ms.calculate_metric("angular_resolution")

        self.assertEqual(0.75, ms.metrics["angular_resolution"]["value"])

    def test_ar081_025(self):
        filename = self.PATH + "test_3_3_AR081_025.graphml"
        ms = MetricsSuite(filename, metrics_list=["angular_resolution"])
        ms.calculate_metric("angular_resolution")

        self.assertAlmostEqual(0.25, ms.metrics["angular_resolution"]["value"], 2)
        self.assertAlmostEqual(0.81, ms.angular_resolution(True), 2)

    def test_eo1(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_orthogonality"])
        ms.calculate_metric("edge_orthogonality")

        self.assertEqual(1, ms.metrics["edge_orthogonality"]["value"])

    def test_eo056(self):
        filename = self.PATH + "test_6_5_EO044.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_orthogonality"])
        ms.calculate_metric("edge_orthogonality")
        
        self.assertAlmostEqual(0.56, ms.metrics["edge_orthogonality"]["value"], 2)

    def test_ec1(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_crossing"])
        ms.calculate_metric("edge_crossing")

        self.assertEqual(1, ms.metrics["edge_crossing"]["value"])

    def test_sym1(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["symmetry"])
        ms.calculate_metric("symmetry")

        self.assertEqual(1, ms.metrics["symmetry"]["value"])

    def test_el1(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_length"])
        ms.calculate_metric("edge_length")

        self.assertEqual(1, ms.metrics["edge_length"]["value"])

    def test_gr1(self):
        filename = self.PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["gabriel_ratio"])
        ms.calculate_metric("gabriel_ratio")

        self.assertEqual(1, ms.metrics["gabriel_ratio"]["value"])


    def test_complete4(self):
        filename = self.PATH + "test_complete4.graphml"
        ms = MetricsSuite(filename)
        self.assertAlmostEqual(0.67, ms.edge_crossing(), 2)
        self.assertAlmostEqual(1, ms.symmetry(), 2)
        self.assertAlmostEqual(0.375, ms.angular_resolution(True), 2)
        self.assertAlmostEqual(0.33, ms.edge_orthogonality(), 2)

        #self.assertAlmostEqual(0.57, ms.node_orthogonality(), 2)





if __name__ == "__main__":
    unittest.main()