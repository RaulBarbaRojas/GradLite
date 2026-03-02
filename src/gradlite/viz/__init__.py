"""Provides clear visualizations of `gradlite` objects, including
backpropagation gradient flow visualizations.
"""

try:
    import graphviz
    del graphviz
except ImportError:
    raise ImportError(
        "gradlite's viz module requires Graphviz. "
        "Please install system Graphviz and pip install grad[viz]"
    )

from gradlite.viz.computational_graph import ComputationalGraphViz

__all__ = ['ComputationalGraphViz']
