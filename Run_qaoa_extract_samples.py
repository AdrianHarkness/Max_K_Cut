import time
from qiskit_optimization.translators import from_docplex_mp # creates QuadraticProgram from docplex model

def run_qaoa_extract_samples(DP_model_list, labels, optimizer):
    """
    Run QAOA for each model in DP_model_list and extract raw QAOA samples.

    For each model, this function:
      - Solves the model classically to obtain the maximum objective value.
      - Converts the docplex model to the required format.
      - Runs QAOA to obtain raw samples.

    Args:
        DP_model_list (list): A list of docplex models.
        labels (list): A list of labels corresponding to each model.

    Returns:
        dict: A dictionary where each key is a label and its value is another dictionary
              containing:
                  - "max_fval": Maximum objective value computed classically.
                  - "samples": Raw QAOA samples.
                  - "dp_model": The original docplex model.
    """
    sample_results = {}
    for dp_model, label in zip(DP_model_list, labels):
        print(f"Solving {label}:")
        print("Classically calculating max objective value...")
        solved_model = dp_model.solve()
        max_fval = solved_model.objective_value
        print(f"Max objective value for {label}: {max_fval}")
        if max_fval == 0:
            raise ValueError("Maximum objective value is zero; cannot compute approximation ratio.")

        # Convert the docplex model to the required format.
        model = from_docplex_mp(dp_model)

        # Run QAOA on the model.
        print("Running QAOA...")
        start_time = time.time()
        qaoa_result = optimizer.solve(model)
        end_time = time.time()
        print("QAOA result:", qaoa_result.prettyprint())
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        samples = qaoa_result.samples

        #only care about approx ratio
        for sample in samples:
            sample.fval = sample.fval / max_fval

        sample_results[label] = {
            "samples": samples,
            "dp_model": dp_model
        }
    return sample_results