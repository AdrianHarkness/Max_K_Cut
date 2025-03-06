import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization.algorithms import OptimizationResultStatus, SolutionSample
import copy

def generate_graph(num_nodes, edge_probability, weighted=True, weight_range=1, seed=None):
    """
    Generate a random graph with weighted edges.

    Args:
        num_nodes (int): Number of nodes in the graph.
        edge_probability (float): Probability of edge creation.
        weighted (bool): Whether to add random weights to the edges.
        weight_variance (int): The range of weights (-weight_variance to weight_variance).

    Returns:
        G (networkx.Graph): The generated graph with weighted edges.
    """
    # Create a random graph

    if seed is None:
        np.random.seed()
    
    G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)

    if weighted:
        # Add random weights to the edges, resampling if the weight is zero
        for (u, v) in G.edges():
            wt = 0
            while wt == 0:
                wt = np.random.randint(-1 * weight_range, weight_range + 1)
                #wt = np.random.uniform(-1 * weight_range, weight_range)
            G.edges[u, v]['weight'] = wt
    else:
        # Add random weights to the edges
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = 1  

    # Assert that no weights are zero
    for (u, v, d) in G.edges(data=True):
        assert d['weight'] != 0, f"Edge ({u}, {v}) has a weight of zero."

    return G


def plot_graph(G):
    """
    Plot the graph with edges colored based on their weights.
    If the graph is unweighted (i.e. all edge weights are equal), a simplified view is shown without a colorbar.
    
    Args:
        G (networkx.Graph): The graph to be plotted.
    """
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Draw the graph positions
    pos = nx.spring_layout(G)  # positions for all nodes

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=10)
    
    # Get all edge data
    edges = list(G.edges(data=True))
    
    # Extract edge weights and check if graph is weighted
    weights = [d['weight'] for (_, _, d) in edges]
    is_weighted = not all(w == weights[0] for w in weights)
    
    if is_weighted:
        # Create a colormap and normalize weights for color mapping
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
        
        # Draw edges with colors based on weights
        for (u, v, d) in edges:
            weight = d['weight']
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2, edge_color=[cmap(norm(weight))])
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge Weight')
    else:
        # For an unweighted graph, simply draw edges in a uniform color.
        nx.draw_networkx_edges(G, pos)

    # Optionally draw labels (commented out for a simplified view)
    # nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)
    
    plt.show()


def expected_value(samples):
    """
    Compute the expected objective value from the sample fvals and probabilities.
    """
    expval = 0
    for sample in samples:
        expval += sample.fval * sample.probability
    return expval

def sample_std(samples, mean_val):
    """
    Compute the standard deviation of the sample fvals around the mean value.
    """
    variance = sum(sample.probability * (sample.fval - mean_val)**2 for sample in samples)
    return np.sqrt(variance)


def get_solution_values(model):
    """
    Solves the given optimization model and retrieves the variable assignments.

    Args:
        model: The optimization model to solve.

    Returns:
        dict or None: A dictionary mapping variable names to their assigned values if a solution is found;
        otherwise, returns None.
    """
    solution = model.solve()
    if solution:
        var_values = {var.name: solution.get_value(var) for var in model.iter_variables()}
        return var_values
    else:
        return None


def rbqo_to_bqo(solution_rbqo, G, K):
    """
    Reconstructs the full variable assignments for the BQO model from the solution of the RBQO model.

    In the RBQO model, the last partition variable for each node is implied and not explicitly represented.
    This function computes the implied variables and returns the full assignment suitable for the BQO model.

    Args:
        solution_rbqo (dict): A dictionary of variable assignments from the RBQO model solution.
        G (networkx.Graph): The graph object.
        K (int): The number of partitions.

    Returns:
        dict: A dictionary representing the full variable assignments, including implied variables, for the BQO model.
    """
    num_nodes = G.number_of_nodes()
    full_assignment_rbqo = {}
    for i in range(num_nodes):
        sum_x = 0
        for j in range(K - 1):  # Adjusted range to start from 0
            var_name = f"x{i}{j}"
            x_i_j = solution_rbqo.get(var_name, 0)
            full_assignment_rbqo[var_name] = x_i_j
            sum_x += x_i_j
        # Implied variable x_i_{K-1}
        x_i_K_minus_1 = 1 - sum_x
        full_assignment_rbqo[f"x{i}{K - 1}"] = x_i_K_minus_1
    return full_assignment_rbqo


def rqubo_to_bqo(solution_rqubo, G, K):
    """
    Reconstructs the full variable assignments for the BQO model from the solution of the RQUBO model.

    In the RQUBO model, the last partition variable for each node is implied and not explicitly represented.
    This function computes the implied variables and returns the full assignment suitable for the BQO model.

    Args:
        solution_rqubo (dict): A dictionary of variable assignments from the RQUBO model solution.
        G (networkx.Graph): The graph object.
        K (int): The number of partitions.

    Returns:
        dict: A dictionary representing the full variable assignments, including implied variables, for the BQO model.
    """
    num_nodes = G.number_of_nodes()
    full_assignment_rqubo = {}
    for i in range(num_nodes):
        sum_x = 0
        for j in range(K - 1):  # Adjusted range to start from 0
            var_name = f"x{i}{j}"
            x_i_j = solution_rqubo.get(var_name, 0)
            full_assignment_rqubo[var_name] = x_i_j
            sum_x += x_i_j
        # Implied variable x_i_{K-1}
        x_i_K_minus_1 = 1 - sum_x
        full_assignment_rqubo[f"x{i}{K - 1}"] = x_i_K_minus_1
    return full_assignment_rqubo


def compute_bqo_objective(assignments, G, K):
    """
    Computes the objective value of the BQO (Binary Quadratic Optimization) model given the variable assignments.

    The objective function calculates the sum over all edges of the weighted difference between 1 and the product
    of the partition assignment variables for connected nodes.

    Args:
        graph (networkx.Graph): The graph object containing the nodes and edges.
        assignments (dict): A dictionary mapping variable names to their assigned values.
        K (int): The number of partitions.

    Returns:
        float: The computed objective value.
    """
    objective_value = 0
    for u, v in G.edges():
        weight = G.edges[u, v]["weight"]
        sum_product = 0
        for j in range(K):  # Adjust range if partitions start from 0
            var_name_u = f"x{u}{j}"
            var_name_v = f"x{v}{j}"
            x_u_j = assignments.get(var_name_u, 0)
            x_v_j = assignments.get(var_name_v, 0)
            sum_product += x_u_j * x_v_j
        objective_value += weight * (1 - sum_product)
    return objective_value

import copy


def feasibility_filter(G, K, samples, label, set_fval_to_zero=False):
    """
    Filter results to keep only feasible solutions and calculate the percentage of feasible solutions.
    
    Args:
        G: Graph object.
        K: Number of partitions or colors.
        samples: List of samples from the optimization result.
        label: String indicating the type of model ("BQO", "RBQO", "QUBO", "RQUBO").
        set_fval_to_zero: Boolean indicating whether to set the fval of infeasible samples to zero.
    
    Returns:
        A tuple containing:
        - A list of feasible samples.
        - The percentage of feasible solutions.
    """
    samples_copy = copy.deepcopy(samples)
    num_nodes = G.number_of_nodes()
    feasible_samples = []

    if label.startswith("QUBO"):
        for sample in samples_copy:
            is_feasible = True
            x = np.array(sample.x).reshape((num_nodes, K))
            for i in range(num_nodes):
                if np.sum(x[i, :]) != 1:
                    is_feasible = False
                    break
            if is_feasible:
                feasible_samples.append(sample)

    elif label.startswith("RQUBO"):
        for sample in samples_copy:
            is_feasible = True
            x = np.array(sample.x).reshape((num_nodes, K - 1))
            for i in range(num_nodes):
                if np.sum(x[i, :]) not in (0, 1):
                    is_feasible = False
                    break
            if is_feasible:
                feasible_samples.append(sample)

    elif label.startswith("BQO"):
        for sample in samples_copy:
            if sample.status == OptimizationResultStatus.SUCCESS:
                feasible_samples.append(sample)

    elif label.startswith("RBQO"):
        for sample in samples_copy:
            if sample.status == OptimizationResultStatus.SUCCESS:
                feasible_samples.append(sample)

    else:
        raise ValueError("Invalid label. Needs to start with 'QUBO', 'RQUBO', 'BQO', or 'RBQO'.")

    # Calculate the percentage of feasible solutions
    total_prob = sum(sample.probability for sample in samples_copy)
    assert np.isclose(total_prob, 1), f"Total probability of all samples is {total_prob}, not 1."
    probability_feasible = sum(sample.probability for sample in feasible_samples)
    
    # Normalize the probabilities of feasible samples
    if probability_feasible > 0:
        for sample in feasible_samples:
            sample.probability /= probability_feasible
        normalized_feasible_prob = sum(sample.probability for sample in feasible_samples)
        assert np.isclose(normalized_feasible_prob, 1), f"Total probability of feasible samples after normalization is {normalized_feasible_prob}, not 1."

    print(f"Probability of feasible solution for {label}: {probability_feasible:.4f}")
    
    if set_fval_to_zero:        
        # Create a set of feasible sample x values for comparison
        feasible_x_set = {tuple(sample.x) for sample in feasible_samples}
        
        # Set infeasible samples' fval to zero in the copy
        for sample in samples_copy:
            if tuple(sample.x) not in feasible_x_set:
                sample.fval = 0
        
        return samples_copy, probability_feasible

    else:
        return feasible_samples, probability_feasible


def repenalize(samples, new_model):
    new_samples = []
    for sample in samples:
        new_sample = SolutionSample(
            x=sample.x,
            fval=sample.fval,  # to be updated below
            probability=sample.probability,
            status=sample.status,
        )
        new_sample.fval = new_model.objective.evaluate(sample.x)
        new_samples.append(new_sample)
    return new_samples