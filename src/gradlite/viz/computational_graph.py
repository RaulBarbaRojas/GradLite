"""Helps visualize the calculated gradients of a given parameter.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphviz import Digraph

from gradlite.core.parameter import Operation, Parameter


@dataclass
class Node:
    """A class to represent nodes in the graph visualization.
    """

    value: Parameter

    def get_label(self) -> str:
        """Gets the label of the node, typically the parameter's label
        whenever available, else its object identifier.

        :return: The node's label.
        """
        node_label = (self.value.label if self.value.label is not None
                      else 'unknown')
        return node_label

    def get_id(self) -> str:
        """Gets the node's identifier based on the object id of the
        param.

        :return: The node's identifier.
        """
        return str(id(self.value))


@dataclass
class Edge:
    """A class to represent edges in the graph visualization.
    """

    source_nodes: list[Node]
    target_node: Node
    operation: Operation

    def get_operation_id(self) -> str:
        """Gets the operation identifier based on the target node's
        identifier and its operation tag.

        :return: The operation identifier.
        """
        return f'{self.operation.value}{self.target_node.get_id()}'

    def get_operation_label(self) -> str:
        """Gets the operation's label.

        :return: The operation's label.
        """
        return self.operation.value


@dataclass
class ComputationalGraphViz:
    """An abstraction of the computational graph of a scalar-valued
    function to be used for visualization purposes.
    """

    nodes: list[Node] = field(default_factory=list[Node])
    edges: list[Edge] = field(default_factory=list[Edge])

    @classmethod
    def from_parameter(cls, parameter: Parameter) -> 'ComputationalGraphViz':
        """Creates a computational graph visualization from a parameter.

        :param parameter: The parameter whose computational graph
        visualization will be created.
        :return: The computational graph visualization object.
        """
        def trace_node(
            parameter: Parameter
        ) -> tuple[list[Node], list[Edge]]:
            """Calculates the trace of a node, including the source
            nodes and corresponding edges.

            :param parameter: The parameter to be traced.
            :return: The params and edges that lead to the given param.
            """
            edges: list[Edge] = []
            nodes: list[Node] = []

            if len(parameter._prev) == 0:
                nodes.append(Node(parameter))
                return nodes, edges

            for source_node in parameter._prev:
                traced_nodes, traced_edges = trace_node(source_node)
                nodes += traced_nodes
                edges += traced_edges

            nodes.append(Node(parameter))
            edges.append(Edge(source_nodes=[Node(parameter)
                                            for parameter in parameter._prev],
                              target_node=Node(parameter),
                              operation=Operation(parameter._op)))

            return nodes, edges

        nodes, edges = trace_node(parameter)
        return cls(nodes, edges)

    def render(self, out_file: str | Path, *args: Any,
               **kwargs: Any) -> Digraph:
        """Renders the visualization graph using `graphviz`.

        :param out_file: The output file of the computational graph viz.
        :return: The `graphviz` directed graph showcasing the
        computional graph of this object.
        """
        digraph = Digraph(graph_attr={'rankdir': 'LR'})

        for node in self.nodes:
            digraph.node(
                name=node.get_id(),
                label=(f'{{{node.get_label()}={node.value.value:.4f} '
                       f'| grad={node.value.grad:.4f}}}'),
                shape='record'
            )

        for edge in self.edges:
            digraph.node(name=edge.get_operation_id(),
                         label=edge.get_operation_label())
            for source_node in edge.source_nodes:
                digraph.edge(source_node.get_id(), edge.get_operation_id())

            digraph.edge(edge.get_operation_id(), edge.target_node.get_id())

        digraph.render(outfile=out_file, *args, **kwargs)
        return digraph
