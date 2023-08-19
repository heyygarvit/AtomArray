from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool, TapTool, BoxSelectTool, Legend, LegendItem

def visualize_graph(nodes):
    # Create a new plot with interactive tools
    p = figure(title="Computational Graph", tools="hover,pan,box_zoom,reset,tap,box_select,wheel_zoom", 
               tooltips=[("Value", "@value"), ("Gradient", "@gradient"), ("Type", "@node_type"), ("Error", "@error")])

    # Extract node data and positions
    x_positions = [node.x for node in nodes]
    y_positions = [node.y for node in nodes]
    values = [node.value for node in nodes]
    gradients = [node.gradient for node in nodes]
    node_types = ['operation' if hasattr(node, 'operation') else 'variable' for node in nodes]
    errors = [node.error if hasattr(node, 'error') else '' for node in nodes]

    source = ColumnDataSource(data=dict(x=x_positions, y=y_positions, value=values, gradient=gradients, node_type=node_types, error=errors))

    # Color nodes based on type
    colors = ["red" if node_type == 'operation' else "blue" for node_type in node_types]

    # Size nodes based on magnitude of values
    sizes = [10 + abs(value) * 5 for value in values]

    # Plot nodes
    r = p.circle('x', 'y', size=sizes, source=source, color=colors, alpha=0.5, legend_field="node_type", selection_color="green", nonselection_fill_alpha=0.2, nonselection_fill_color="grey")

    # Plot edges with varying thickness based on gradient magnitude
    for node in nodes:
        for parent in node.parents:
            gradient_magnitude = abs(node.gradient)
            p.line([parent.x, node.x], [parent.y, node.y], line_width=2 + gradient_magnitude * 2, alpha=0.6)

    # Interactive legend
    legend = Legend(items=[
        LegendItem(label="operation", renderers=[r]),
        LegendItem(label="variable", renderers=[r])
    ])
    p.add_layout(legend)
    p.legend.click_policy = "hide"

    show(p)
