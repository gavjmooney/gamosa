# Adapted from: https://github.com/hadim/pygraphml/blob/master/pygraphml/graphml_parser.py
from xml.dom import minidom

def write_graphml_pos(graph, filename):

    doc = minidom.Document()
    # create root elements
    root = doc.createElement('graphml')
    root.setAttribute("xmlns", "http://graphml.graphdrawing.org/xmlns")
    root.setAttribute("xmlns:y", "http://www.yworks.com/xml/graphml")
    root.setAttribute("xmlns:yed", "http://www.yworks.com/xml/yed/3")

    doc.appendChild(root)

    # create key attribute for nodegraphics
    attr_node = doc.createElement('key')
    attr_node.setAttribute('id', 'd1')
    attr_node.setAttribute('yfiles.type', "nodegraphics")
    attr_node.setAttribute('for', "node")
    root.appendChild(attr_node)

    # create graph attribute for edges and nodes to be added to
    graph_node = doc.createElement('graph')
    graph_node.setAttribute('id', 'G')
    graph_node.setAttribute('edgedefault', 'undirected')
    root.appendChild(graph_node)

    # Add nodes
    for n in graph.nodes():
        
        node = doc.createElement('node')
        node.setAttribute('id', str(n))
        data = doc.createElement('data')
        data.setAttribute('key', 'd1')

        # Adding node that allows styles and attributes to be added to nodes
        shapeElement = doc.createElement('y:ShapeNode')

        # Set shape of node 
        nodeShape = doc.createElement('y:Shape')
        shape = graph.nodes[n].get("shape_type", 'ellipse')
        nodeShape.setAttribute('type', shape)

        # adding label to node
        nodeLabel = doc.createElement('y:NodeLabel')
        nodeLabel.setAttribute('textColor', '#000000')
        nodeLabel.setAttribute('fontSize', '6')
        label = doc.createTextNode(str(graph.nodes[n].get("label", '\n')))
        nodeLabel.appendChild(label)

        # assign colours to nodes
        nodeColour = doc.createElement('y:Fill')
        nodeColour.setAttribute('transparent', 'false')
        nodeColour.setAttribute('color', "#FFCC00")

        # set size of nodes
        pos = doc.createElement('y:Geometry')
        pos.setAttribute('height', '30')
        pos.setAttribute('width', '30')

        # set x and y coordinates for each point
        pos.setAttribute('x', str(graph.nodes[n].get("x", '0')))
        pos.setAttribute('y', str(graph.nodes[n].get("y", '0')))

        # adding styling attributes to each node
        shapeElement.appendChild(nodeColour)
        shapeElement.appendChild(pos)
        shapeElement.appendChild(nodeShape)
        shapeElement.appendChild(nodeLabel)
        data.appendChild(shapeElement)
        node.appendChild(data)
        graph_node.appendChild(node)

    # Add eges between nodes
    for e in graph.edges():
        edge = doc.createElement('edge')
        edge.setAttribute('source', str(e[0]))
        edge.setAttribute('target', str(e[1]))
        graph_node.appendChild(edge)

    with open(filename, 'w') as f:
        f.write(doc.toprettyxml(indent='    '))
