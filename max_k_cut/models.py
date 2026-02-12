from docplex.mp.model import Model

#BQO Model
# ======================================================================================================================
def docplex_BQO(graph, k, name):
    """
    Create a Binary Quadratic Optimization (BQO) model for the Max-K-Cut problem.
    """
    # Create the model
    model = Model(name=name)

    # Number of vertices
    nodes = graph.number_of_nodes()
    partitions = k

    # Decision variables x_ic (binary) where x_ic = 1 if vertex i is assigned to partition c
    x = {(i, j): model.binary_var(name=f"x{i}{j}")
          for i in range(nodes) for j in range(partitions)}

    # Objective: Maximize the sum of weights of edges between different partitions
    objective_terms = []
    for u, v in graph.edges():
        weight = graph.edges[u, v]['weight']
        inner_product = model.sum(x[(u, j)] * x[(v, j)] for j in range(partitions)) #inner product between rows of x, zero if diff partitions
        objective_terms.append(weight * (1 - inner_product))

    # Constraints: Each vertex must be assigned to exactly one partition
    for v in range(nodes):
        model.add_constraint(model.sum(x[(v, j)] for j in range(partitions)) == 1, f"vertex {v}")

    # Set the objective function to maximize
    model.maximize(model.sum(objective_terms))

    return model


#QUBO Model
# ======================================================================================================================
def docplex_QUBO(graph, k, penalty, name):
    """
    Create a Quadratic Unconstrained Binary Optimization (QUBO) model for the Max-K-Cut problem.
    """
    # Create the model
    model = Model(name=name)

    # Number of vertices
    nodes = graph.number_of_nodes()
    partitions = k

    # Decision variables x_ic (binary) where x_ic = 1 if vertex i is assigned to partition c
    x = {(i, j): model.binary_var(name=f"x{i}{j}") 
         for i in range(nodes) for j in range(partitions)}

    # Objective: Maximize the sum of weights of edges between different partitions
    objective_terms = []
    for u, v in graph.edges():
        weight = graph.edges[u, v]['weight']
        inner_product = model.sum(x[(u, j)] * x[(v, j)] for j in range(partitions))
        objective_terms.append(weight * (1 - inner_product))

    penalty_terms = []
    for v in range(nodes):
        penalty_terms.append(penalty[v] * (model.sum(x[(v, j)] for j in range(partitions)) - 1)**2)

    # Set the objective function to maximize
    model.maximize(model.sum(objective_terms) - model.sum(penalty_terms))

    return model

#QUBO Model with no constraints (meant to be enforced with a mixer instead)
# ======================================================================================================================
def docplex_QUBO_no_constraints(graph, k, name):
    """
    Create a Quadratic Unconstrained Binary Optimization (QUBO) model for the Max-K-Cut problem with no constraints.
    The constraints are meant to be enforced with a mixer instead of penalties.
    """
    # Create the model
    model = Model(name=name)

    # Number of vertices
    nodes = graph.number_of_nodes()
    partitions = k

    # Decision variables x_ic (binary) where x_ic = 1 if vertex i is assigned to partition c
    x = {(i, j): model.binary_var(name=f"x{i}{j}") 
         for i in range(nodes) for j in range(partitions)}

    # Objective: Maximize the sum of weights of edges between different partitions
    objective_terms = []
    for u, v in graph.edges():
        weight = graph.edges[u, v]['weight']
        inner_product = model.sum(x[(u, j)] * x[(v, j)] for j in range(partitions))
        objective_terms.append(weight * (1 - inner_product))

    # Set the objective function to maximize
    model.maximize(model.sum(objective_terms))

    return model


#RBQO Model
# ======================================================================================================================
def docplex_RBQO(graph, k, name):
    """
    Create a Reduced Binary Quadratic Optimization (RBQO) model for the Max-K-Cut problem.
    """
    # Create the model
    model = Model(name=name)

    # Number of vertices
    nodes = graph.number_of_nodes()
    partitions = k

    # Decision variables x_ic (binary) where x_ic = 1 if vertex i is assigned to partition c
    x = {(i, j): model.binary_var(name=f"x{i}{j}") for i in range(nodes) for j in range(partitions-1)}

    # Objective: Maximize the sum of weights of edges between different partitions
    objective_terms = []
    for u, v in graph.edges():
        weight = graph.edges[u, v]['weight']
        x_uk = 1 - model.sum(x[(u, j)] for j in range(partitions-1))
        x_vk = 1 - model.sum(x[(v, j)] for j in range(partitions-1))
        inner_product = model.sum(x[(u, j)] * x[(v, j)] for j in range(partitions-1)) + x_uk*x_vk
        objective_terms.append(weight * (1 - inner_product))
 
    # Constraints: Each vertex must be assigned to exactly one partition
    for v in range(nodes):
        model.add_constraint(model.sum(x[(v, j)] for j in range(partitions-1)) <= 1, f"vertex {v}")
    
    # Set the objective function to maximize
    model.maximize(model.sum(objective_terms))

    return model


#RQUBO Model
# ======================================================================================================================
def docplex_RQUBO(graph, k, penalty, name):
    """
    Create a Reduced Quadratic Unconstrained Binary Optimization (RQUBO) model for the Max-K-Cut problem.
    """
    # Create the model
    model = Model(name=name)

    # Number of vertices
    nodes = graph.number_of_nodes()
    partitions = k

    # Decision variables x_ic (binary) where x_ic = 1 if vertex i is assigned to partition c
    x = {(i, j): model.binary_var(name=f"x{i}{j}") for i in range(nodes) for j in range(partitions-1)}

    # Objective: Maximize the sum of weights of edges between different partitions
    objective_terms = []
    for u, v in graph.edges():
        weight = graph.edges[u, v]['weight']
        x_uk = 1 - model.sum(x[(u, j)] for j in range(partitions-1))
        x_vk = 1 - model.sum(x[(v, j)] for j in range(partitions-1))
        inner_product = model.sum(x[(u, j)] * x[(v, j)] for j in range(partitions-1)) + x_uk*x_vk
        objective_terms.append(weight * (1-inner_product))
    
    penalty_terms = []
    for v in range(nodes):
        penalty_terms.append(penalty[v] * (model.sum((x[(v, i)]*x[(v,j)]) for i in range(partitions-1) for j in range(i+1,partitions-1))))

    # Set the objective function to maximize
    model.maximize(model.sum(objective_terms) - model.sum(penalty_terms))

    return model
