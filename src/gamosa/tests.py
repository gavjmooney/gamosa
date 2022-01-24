from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np
from os.path import exists

import unittest

class TestMetricsSuiteGeneral(unittest.TestCase):


    def test_load(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
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
        filename = PATH + "write_test.graphml"
        ms.write_graph(filename, scale=False)
        self.assertTrue(exists(filename))

        ms = MetricsSuite(filename)
        self.assertEqual(8, ms.graph.number_of_nodes())
        self.assertEqual(10, ms.graph.number_of_edges())
        node_positions = ((v["x"], v["y"]) for k,v in ms.graph.nodes(data=True))
        known_positions = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
        
        for pos in node_positions:
            self.assertIn(pos, known_positions)


class TestMetricsSuiteEdgeCrossing(unittest.TestCase):

    def test_ec1(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_crossing"])
        ms.calculate_metric("edge_crossing")

        self.assertEqual(1, ms.metrics["edge_crossing"]["value"])
        self.assertEqual(0, ms.metrics["edge_crossing"]["num_crossings"])

    def test_ec0(self):
        filename = PATH + "test_4_4_EC0_CA1.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_crossing"])
        ms.calculate_metric("edge_crossing")

        self.assertEqual(0, ms.metrics["edge_crossing"]["value"])
        self.assertEqual(1, ms.metrics["edge_crossing"]["num_crossings"])

    def test_ec06(self):
        filename = PATH + "test_8_7_EC06.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_crossing"])
        ms.calculate_metric("edge_crossing")

        self.assertEqual(0.6, ms.metrics["edge_crossing"]["value"])
        self.assertEqual(6, ms.metrics["edge_crossing"]["num_crossings"])


class TestMetricsSuiteAngularResolution(unittest.TestCase):

    def test_ar1(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["angular_resolution"])
        ms.calculate_metric("angular_resolution")

        self.assertEqual(1, ms.metrics["angular_resolution"]["value"])

    def test_ar075(self):
        filename = PATH + "test_6_5_AR075.graphml"
        ms = MetricsSuite(filename, metrics_list=["angular_resolution"])
        ms.calculate_metric("angular_resolution")

        self.assertEqual(0.75, ms.metrics["angular_resolution"]["value"])

    def test_ar081_025(self):
        filename = PATH + "test_3_3_AR081_025.graphml"
        ms = MetricsSuite(filename, metrics_list=["angular_resolution"])
        ms.calculate_metric("angular_resolution")

        self.assertAlmostEqual(0.25, ms.metrics["angular_resolution"]["value"], 2)
        self.assertAlmostEqual(0.81, ms.angular_resolution(True), 2)


class TestMetricsSuiteNodeResolution(unittest.TestCase):

    def test_nr_055(self):
        filename = PATH + "test_3_2_NR055.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_resolution"])
        ms.calculate_metric("node_resolution")

        self.assertAlmostEqual(0.55, ms.metrics["node_resolution"]["value"], 2)

    def test_nr_1(self):
        filename = PATH + "test_3_3_NR1.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_resolution"])
        ms.calculate_metric("node_resolution")

        self.assertAlmostEqual(1, ms.metrics["node_resolution"]["value"], 2)

    def test_nr_05(self):
        filename = PATH + "test_3_2_NR05.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_resolution"])
        ms.calculate_metric("node_resolution")

        self.assertAlmostEqual(0.5, ms.metrics["node_resolution"]["value"], 2)

    def test_nr_0217(self):
        filename = PATH + "test_3_2_NR0217.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_resolution"])
        ms.calculate_metric("node_resolution")

        self.assertAlmostEqual(0.217, ms.metrics["node_resolution"]["value"], 2)       

class TestMetricsSuiteEdgeLength(unittest.TestCase):

    def test_el1(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_length"])
        ms.calculate_metric("edge_length")

        self.assertEqual(1, ms.metrics["edge_length"]["value"])

    def test_el1_2(self):
        filename = PATH + "test_4_3_NO1_EL1.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_length"])
        ms.calculate_metric("edge_length")

        self.assertEqual(1, ms.metrics["edge_length"]["value"])
    
    def test_el067(self):
        filename = PATH + "test_4_3_NO67_EL067_GR083.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_length"])
        ms.calculate_metric("edge_length")

        self.assertAlmostEqual(0.67, ms.metrics["edge_length"]["value"], 2)


class TestMetricsSuiteSymmetry(unittest.TestCase):

    def test_sym1(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["symmetry"])
        ms.calculate_metric("symmetry")

        self.assertEqual(1, ms.metrics["symmetry"]["value"])

    def test_sym0625(self):
        filename = PATH + "test_10_10_SYM0625.graphml"
        ms = MetricsSuite(filename, metrics_list=["symmetry"])
        ms.calculate_metric("symmetry")

        self.assertAlmostEqual(0.625, ms.metrics["symmetry"]["value"], 3)

    def test_sym_tolerance(self):
        filename = PATH + "test_4_4_SYM025-075.graphml"
        ms = MetricsSuite(filename, metrics_list=["symmetry"])

        # Total area of graph is 17250
        # Area of Equilateral triangle is 4330

        self.assertEqual(0, ms.symmetry())
        # Threshold set to 1 as a triangle when split by bisector will only have one symmetry at each axis
        self.assertAlmostEqual(0.251, ms.symmetry(threshold=1), 3) # 4330 / 17250 ~= 0.251
        # Tolerance set to 1 as an equilateral triangle cannot be represented by three vertexes without at least one irrational vertex
        self.assertAlmostEqual(0.753, ms.symmetry(threshold=1, tolerance=1), 3) # 4330 * 3 / 17250 ~= 0.753


class TestMetricsSuiteEdgeOrthogonality(unittest.TestCase):

    def test_eo1(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_orthogonality"])
        ms.calculate_metric("edge_orthogonality")

        self.assertEqual(1, ms.metrics["edge_orthogonality"]["value"])

    def test_eo056(self):
        filename = PATH + "test_6_5_EO044.graphml"
        ms = MetricsSuite(filename, metrics_list=["edge_orthogonality"])
        ms.calculate_metric("edge_orthogonality")
        
        self.assertAlmostEqual(0.56, ms.metrics["edge_orthogonality"]["value"], 2)


class TestMetricsSuiteNodeOrthogonality(unittest.TestCase):

    def test_no05(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_orthogonality"])
        ms.calculate_metric("node_orthogonality")

        self.assertEqual(0.5, ms.metrics["node_orthogonality"]["value"])

    def test_no05(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_orthogonality"])
        ms.calculate_metric("node_orthogonality")

        self.assertEqual(0.5, ms.metrics["node_orthogonality"]["value"])

    def test_no05_2(self):
        filename = PATH + "test_6_5_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_orthogonality"])
        ms.calculate_metric("node_orthogonality")

        self.assertEqual(0.5, ms.metrics["node_orthogonality"]["value"])

    def test_no1(self):
        filename = PATH + "test_4_3_NO1_EL1.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_orthogonality"])
        ms.calculate_metric("node_orthogonality")

        self.assertEqual(1, ms.metrics["node_orthogonality"]["value"])

    def test_no067(self):
        filename = PATH + "test_4_3_NO67_EL067_GR083.graphml"
        ms = MetricsSuite(filename, metrics_list=["node_orthogonality"])
        ms.calculate_metric("node_orthogonality")

        self.assertAlmostEqual(0.67, ms.metrics["node_orthogonality"]["value"], 2)


class TestMetricsSuiteGabrielRatio(unittest.TestCase):

    def test_gr1(self):
        filename = PATH + "test_6_5_AR1_EO1_EC1_SYM1_EL1_GR1_NO05.graphml"
        ms = MetricsSuite(filename, metrics_list=["gabriel_ratio"])
        ms.calculate_metric("gabriel_ratio")

        self.assertEqual(1, ms.metrics["gabriel_ratio"]["value"])

    def test_gr083(self):
        filename = PATH + "test_4_3_NO67_EL067_GR083.graphml"
        ms = MetricsSuite(filename, metrics_list=["gabriel_ratio"])
        ms.calculate_metric("gabriel_ratio")

        self.assertAlmostEqual(0.83, ms.metrics["gabriel_ratio"]["value"], 2)


class TestMetricsSuiteCrossingAngle(unittest.TestCase):
    
    def test_ca1(self):
        filename = PATH + "test_4_4_EC0_CA1.graphml"
        ms = MetricsSuite(filename, metrics_list=["crossing_angle"])
        ms.calculate_metric("crossing_angle")

        self.assertEqual(1, ms.metrics["crossing_angle"]["value"])

    def test_ca2(self):
        filename = PATH + "test_7_7_CA1.graphml"
        ms = MetricsSuite(filename, metrics_list=["crossing_angle"])
        ms.calculate_metric("crossing_angle")

        self.assertEqual(1, ms.metrics["crossing_angle"]["value"])

    def test_ca05(self):
        filename = PATH + "test_4_3_CA05.graphml"
        ms = MetricsSuite(filename, metrics_list=["crossing_angle"])
        ms.calculate_metric("crossing_angle")

        self.assertEqual(0.5, ms.metrics["crossing_angle"]["value"])

    def test_ca057(self):
        filename = PATH + "test_4_3_CA057.graphml"
        ms = MetricsSuite(filename, metrics_list=["crossing_angle"])
        ms.calculate_metric("crossing_angle")

        self.assertAlmostEqual(0.57, ms.metrics["crossing_angle"]["value"], 2)



class TestMetricsSuiteCombinations(unittest.TestCase):

    def test_complete4(self):
        filename = PATH + "test_complete4.graphml"
        ms = MetricsSuite(filename)
        self.assertAlmostEqual(0.67, ms.edge_crossing(), 2)
        self.assertAlmostEqual(1, ms.symmetry(), 2)
        self.assertAlmostEqual(0.375, ms.angular_resolution(True), 2)
        self.assertAlmostEqual(0.33, ms.edge_orthogonality(), 2)

        #self.assertAlmostEqual(0.57, ms.node_orthogonality(), 2)


if __name__ == "__main__":
    PATH = "..\\..\\graphs\\moon\\"
    unittest.main()