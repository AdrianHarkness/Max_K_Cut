from qiskit_optimization.translators import from_docplex_mp # creates QuadraticProgram from docplex model
from Helper_Functions import repenalize, feasibility_filter
import numpy as np
import matplotlib.pyplot as plt

def plot_qaoa_histogram(
    sample_results,
    *keys,
    Graph=None,
    Partitions=None,
    filter_infeasible=False,
    new_pen_list=None,
    bins='auto'
):
    """
    Plot a histogram of probability vs. approximation ratio for QAOA results using plt.hist.

    This function processes the raw QAOA samples by optionally re-penalizing and/or filtering them,
    then plots a histogram for each provided key in sample_results.

    If keys are provided (after sample_results), only those specified keys
    will be plotted. For example:
        plot_qaoa_histogram(results_dict, 'QUBO (Tight)', 'RQUBO (Tight)')

    Args:
        sample_results (dict): A dictionary where each key is a label and its value is a dict containing:
            - "samples": Raw QAOA samples.
            - "dp_model": The original docplex model.
        *keys (str): Optional keys specifying which results to plot.
        Graph (optional): Graph object for feasibility filtering.
        Partitions (optional): Partitions for feasibility filtering.
        filter_infeasible (bool, optional): If True, filter out infeasible solutions.
            Requires Graph and Partitions to be provided.
        new_pen_list (dict, optional): A dictionary mapping each label to a boolean indicating
            whether to re-penalize the samples. If None, defaults to all False.
        bins: Either an integer (or array of bin edges) or a string specifying the binning rule.
              If it's a string (e.g. 'auto'), we compute a common set of bin edges from all data.
        
    Returns:
        None
    """
    # If keys are provided, subset sample_results accordingly.
    if keys:
        sample_results = {k: sample_results[k] for k in keys if k in sample_results}
    
    # Set default re-penalization flags if not provided.
    if new_pen_list is None:
        new_pen_list = {label: False for label in sample_results}
    
    if filter_infeasible and (Graph is None or Partitions is None):
        raise ValueError("Graph and Partitions must be provided when filter_infeasible is True.")
    
    # # Mapping for re-penalization: keys match the label exactly.
    # repen_mapping = {
    #     "QUBO (Tight)": dp_qubo_naive,
    #     "QUBO (Naive)": dp_qubo_tight,
    #     "RQUBO (Tight)": dp_rqubo_naive,
    #     "RQUBO (Naive)": dp_rqubo_tight,
    # }

    # Prepare to collect data for all histograms (so we can compute common bin edges).
    processed_hist_data = []

    # Process each label (entry) in sample_results.
    for label, result in sample_results.items():
        samples = result["samples"]
        current_label = label
        
        # # If re-penalization is requested, look up this label in repen_mapping:
        # if new_pen_list.get(label, False):
        #     if label not in repen_mapping:
        #         raise ValueError(f"Label '{label}' not recognized for re-penalization mapping.")
        #     new_model = from_docplex_mp(repen_mapping[label])
        #     samples = repenalize(samples, new_model)
        #     current_label += " (Repenalized)"
        
        # Filter infeasible samples if requested.
        if filter_infeasible:
            samples, _ = feasibility_filter(Graph, Partitions, samples, current_label)
            # Normalize probabilities
            total_prob = sum(sample.probability for sample in samples)
            for sample in samples:
                sample.probability /= total_prob
            
        # Compute approximation ratios and probabilities for this dataset.
        approx_ratios = [sample.fval for sample in samples] #already normalized in previous function
        probs = [sample.probability for sample in samples]
        
        # Store them for plotting later (all at once).
        processed_hist_data.append((approx_ratios, probs, current_label))
    
    # If bins is a string (e.g. 'auto'), we compute a common set of bin edges
    # from all approximation ratio data so that every dataset uses the same bins.
    
    all_approx = []
    for approx_ratios, _, _ in processed_hist_data:
        all_approx.extend(approx_ratios)
    
    if isinstance(bins, str):
        bin_edges = np.histogram_bin_edges(all_approx, bins=bins)
    else:
        bin_edges = bins

    # Create the plot
    plt.figure(figsize=(8, 6))

    for approx_ratios, probs, current_label in processed_hist_data:
        n, bins, patches = plt.hist(
            approx_ratios,
            bins=bin_edges,
            weights=probs,
            alpha=0.35,
            label=current_label,
            histtype='stepfilled'
        )
        
        # Extract face color from the first patch
        color = patches[0].get_facecolor()
        
        # Compute the expected value
        exp_val = np.sum(np.array(approx_ratios) * np.array(probs))
        
        # Plot a vertical line in that color
        plt.vlines(
            exp_val, 0, 1, linestyles='dashed', color=color, lw=1,
            label=f"Expected Value of {current_label}: {exp_val:.2f}"
        )

    plt.xlabel('Approximation Ratio')
    plt.ylabel('Probability')
    title = 'QAOA Distribution'
    if filter_infeasible:
        title += ' (Feasible Solutions Only)'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()