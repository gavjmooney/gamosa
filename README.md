# gamosa
Graph Aesthetic Metric Optimization by Simulated Annealing. A suite of functions for calculating several aesthetic metrics of a given graph drawing, with a simulated annealing algorithm for creating a new drawing which optimizes these metrics.

## Installation
To install, download the repository and create a Python3 virtual environment (https://docs.python.org/3/library/venv.html) in the top level of the directory. Activate the virtual environment and enter the command 'python3 pip install -r requirements.txt'. (Assuming python3 is on the system's PATH environment variable). Files can then be executed by first changing to the src\gamosa\ directory and entering 'python3 [filename]'.


## Files
metrics_suite.py contains the implementation of several metrics for graph drawing aesthetics. It makes use of write_graph.py for saving files to GraphML format.

tests.py contains several unit tests for checking the correctness of the metrics implemented in metrics_suite.py. The graph drawings used in these tests are found in graphs/test_graphs/.

simulated_annealing.py contains an implementation of a simulated annealing algorithm which aims to draw graphs by starting with an initial layout and improving one or more aesthetic metrics. It has options for several initial graph layouts, cooling schedules, and creation of new layouts. The simulated_annealing algorithm uses a MetricSuite object as the basis of its cost function.

main.py contains a basic example script which calculates the metric values for a directory of graph drawings and outputs them to a csv file.

correlations.py, distributions,py, evaluation.py, and experiment.py contain functions used as part of research into graph metrics and simulated annealing and will likely be moved out of this repository.


## Use
main.py can be used as a driver script for the metric_suite.py and simulated_annealing.py files. The script contains examples using the code.