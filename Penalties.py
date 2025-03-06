import numpy as np

# Penalties for the BQO constraints
#================================================================================================
def tight_qubo_penalty(graph, k):
    c = np.zeros(graph.number_of_nodes())
    for v in graph.nodes():
        neighbors_plus = []
        neighbors_minus = []
        #sort neighbors into positive and negative weights
        for u in graph.neighbors(v):
            if graph.edges[v, u]['weight'] > 0:
                neighbors_plus.append(u)
            else:
                neighbors_minus.append(u)
        dv_plus = sum([graph.edges[v, u]['weight'] for u in neighbors_plus])
        dv_minus = sum([graph.edges[v, u]['weight'] for u in neighbors_minus])
        c[v] = max(dv_plus/k, -1*dv_minus/2) #from paper.  Is it -3/2 or -1/2???
    return c

def naive_qubo_penalty(graph, k):
    
    #c = 1e2*np.ones(graph.number_of_nodes())

    #c = 10 * max(tight_qubo_penalty(graph, k))*np.ones(graph.number_of_nodes())

    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))
    num_nodes = graph.number_of_nodes()
    c = max(num_nodes/k, total_weight)*np.ones(num_nodes)

    return c

def interpolated_qubo_penalty(graph, k, t):
    """
    Interpolated penalty function between tight and naive penalties.
    t = 0 -> naive penalty
    t = 1 -> tight penalty
    """
    if t < 0 or t > 1:
        raise ValueError("t must be between 0 and 1.")
    
    tight_pen = tight_qubo_penalty(graph, k)
    naive_pen = naive_qubo_penalty(graph, k)

    return t * tight_pen + (1-t) * naive_pen



# Penalties for the R-BQO constraints
#================================================================================================
def tight_rqubo_penalty(graph, k):
    c = np.zeros(graph.number_of_nodes())
    for v in graph.nodes():
        neighbors_plus = []
        neighbors_minus = []
        #sort neighbors into positive and negative weights
        for u in graph.neighbors(v):
            if graph.edges[v, u]['weight'] > 0:
                neighbors_plus.append(u)
            else:
                neighbors_minus.append(u)
        dv_plus = sum([graph.edges[v, u]['weight'] for u in neighbors_plus])
        dv_minus = sum([graph.edges[v, u]['weight'] for u in neighbors_minus])
        c[v] = (dv_plus - dv_minus) # is it - 2 dv_minus or - dv_minus???
    return c

def naive_rqubo_penalty(graph, k):
    
    #c = 1e2*np.ones(graph.number_of_nodes())
    
    #c = 10 * max(tight_rqubo_penalty(graph, k))*np.ones(graph.number_of_nodes())

    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))
    num_nodes = graph.number_of_nodes()
    c = max(num_nodes/k, total_weight)*np.ones(num_nodes)
    
    return c

def interpolated_rqubo_penalty(graph, k, t):
    '''
    Interpolated penalty function between tight and naive R-BQO penalties.
    t = 0 -> naive penalty
    t = 1 -> tight penalty
    '''
    if t < 0 or t > 1:
        raise ValueError("t must be between 0 and 1.")
    
    tight_pen = tight_rqubo_penalty(graph, k)
    naive_pen = naive_rqubo_penalty(graph, k)

    return t * tight_pen + (1-t) * naive_pen