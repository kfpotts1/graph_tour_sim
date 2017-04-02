import numpy as np
cimport numpy as np

# np.import_array()

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
def _tour_graph(np.ndarray[DTYPE_t, ndim=2] g not None, int start=0, str start_method = 'int', int MAX_ITER =-1, int seed=-1):
    """
    Tours a graph (random walk) and returns tour time and tour stops.
    
    :param g: (np.ndarray[DTYPE_t, ndim=2])
        adjacency matrix for the graph to be toured
    :param start: (int >= 0) defaults to 0
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

    cdef int num_vertices = g.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] visited = np.zeros(g.shape[0], dtype=DTYPE)

    # from my reading, it seams dict will be best here
    cdef dict visit_dict = dict()

    # implement method for choosing start
    if start_method == 'uniform_random':
        start = np.random.randint(0,high=g.shape[0],size=1, dtype=DTYPE)

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

def tour_graph(np.ndarray graph , int start=0, str start_method = 'int', int MAX_ITER =-1, int seed=-1):
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
    g = np.asarray(graph, dtype=DTYPE)
    return _tour_graph(g, start=start, start_method=start_method, MAX_ITER=MAX_ITER, seed=seed)


@cython.boundscheck(False) # turn of bounds-checking for entire function
def _run_simulation(np.ndarray[DTYPE_t, ndim=2] graph not None, int n=1000, int start=0, str start_method = 'int', int MAX_ITER =-1):
    """
    Runs n tours on the given graph and returns the tour time for each tour.
    
    :param graph: (np.ndarray[DTYPE_t, ndim=2])
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
    cdef np.ndarray[DTYPE_t, ndim=1] data = np.zeros(n, dtype=DTYPE)
    cdef int i
    for i in range(n):
        data[i] = tour_graph(graph, start=start, start_method=start_method, MAX_ITER=MAX_ITER)[0]
    return data

def run_simulation(np.ndarray graph, int n=1000, int start=0, str start_method = 'int', int MAX_ITER =-1):
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
    graph = np.asarray(graph, dtype=DTYPE)
    return _run_simulation(graph, n=n, start=start, start_method=start_method, MAX_ITER=MAX_ITER)