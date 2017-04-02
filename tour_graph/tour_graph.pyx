import numpy as np
cimport numpy as np

def tour_graph(np.ndarray graph, int start=0, str start_method = 'int', int MAX_ITER =-1, int seed=-1):
    """
    Tours a graph (random walk) and returns tour time and tour stops.
    
    :param graph: (array_like)
        adjacency matrix for the graph to be toured
    :param start: (int) defaults to 0
        the node at which to start the tour
    :param start_method: (str) defaults to 'int'
        the method used to choose a starting node
        'int': indicates that the start (int) argument should be used as the first node
        'uniform_random': will select the first node at random from all nodes on the graph
    :param MAX_ITER: (int) defaults to -1
        maximum number of stops on the tour
        -1: for no maximum
    :param seed: (int) defaults to -1
        sets the seed for the numpy prng
        -1 for no seed
    :return:
        stops: (int)
            the number of stops on the tour
        visits: (dict)
            dictionary with nodes as keys, and lists of the stop times for each node
    """

    # set seed for prng
    if seed != -1:
        np.random.seed(seed)

    cdef np.ndarray g = np.asarray(graph)
    cdef int num_vertices = g.shape[0]

    cdef np.ndarray visited = np.zeros(g.shape[0], dtype=np.int)

    # from my reading, it seams dict will be best here
    cdef dict visit_dict = dict()

    # implement method for choosing start
    if start_method == 'uniform_random':
        start = np.random.randint(0,high=g.shape[0],size=1)

    cdef int v = start
    visited[v] = 1 # make the first stop
    cdef int counter = 0

    while visited.sum() != num_vertices:
        counter += 1
        v = np.random.choice(np.arange(g.shape[0])[g[v,:] > 0])
        visited[v] = 1
        if v in visit_dict.keys():
            visit_dict[v].append(counter)
        else:
            visit_dict[v] = [counter]
        if MAX_ITER != -1:
            if counter >=MAX_ITER:
                raise StopIteration('number of iterations has reached specified MAX_ITER.')

    return counter, visit_dict


def run_simulation( np.ndarray graph, int n=1000, int start=0, str start_method = 'int', int MAX_ITER =-1):
    """
    Runs n tours on the given graph and returns the tour time for each tour.
    
    :param graph: (array_like)
        adjacency matrix for the graph to be toured
    :param n: (int)
        number of tours in simulation
    :param start: (int) defaults to 0
        the node at which to start the tour
    :param start_method: (str) defaults to 'int'
        the method used to choose a starting node
        'int': indicates that the start (int) argument should be used as the first node
        'uniform_random': will select the first node at random from all nodes on the graph
    :param MAX_ITER: (int) defaults to -1
        maximum number of stops on the tour
        -1: for no maximum
    :return: (numpy array) tour times for each tour
    """
    cdef np.ndarray data = np.zeros(n)
    cdef int i
    for i in range(n):
        data[i] = tour_graph(graph, start=start, start_method=start_method, MAX_ITER=MAX_ITER)[0]
    return data
