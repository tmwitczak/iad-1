# /////////////////////////////////////////////////////////////////// Imports #
import random
import statistics
from typing import List, Tuple

import math
import numpy
from matplotlib import pyplot

from src.file_reading import ClusteringData, DataMode, Vector, vector_from_list


# ///////////////////////////////////////////////////////// Kohonen algorithm #
def kohonen(
        data_set: ClusteringData,
        n: int,
        *,
        iterations: int = 10,
        animation_rate: float = 0.001,
        x_axis_vector_index: int = 0,
        y_axis_vector_index: int = 1,
        mode: DataMode = DataMode.DEFAULT) \
        -> None:
    """ Compute self-organising map using Kohonen network algorithm.

    Parameters
    ----------
    animation_rate : float
        # TODO
    data_set : ClusteringData
        Data set for finding clusters.
    n : int
        Number of neurons in Kohonen self-organising map.
    iterations
        Number of times the algorithm will be executed, the result with the
        lowest quantisation error will be chosen.
    x_axis_vector_index : int
        Index of vector element to plot on x axis.
    y_axis_vector_index : int
        Index of vector element to plot on y axis.
    mode : KohonenDataMode
        Select whether data is normalised or standardised

    """
    # ----------------------------------------------------------------------- #
    # Choose appropriate data mode
    vectors: Tuple[Vector, ...] = get_vectors_from_data_set(data_set, mode)

    # Perform a given number of iterations of Kohonen algorithm
    # and choose the result with the lowest quantisation error
    initial_neurons: List[Tuple[int, ...]] = []
    quantisation_errors: List[float] = []

    print('[ Kohonen Network ]')

    for i in range(iterations):
        print_status_bar('> Iterations', i + 1, iterations)

        # Select initial neurons as n random vectors from data set
        initial_neurons.append(
                pick_random_neurons(n, vectors))

        # Perform Kohonen algorithm and save quantisation error
        quantisation_errors.append(
                kohonen_iteration(vectors, initial_neurons[i],
                                  data_set.classes, data_set.parameter_names))

    initial_neurons_with_lowest_error: Tuple[int, ...] \
        = initial_neurons[quantisation_errors.index(min(quantisation_errors))]

    # Draw animation for the best iteration
    kohonen_iteration(vectors, initial_neurons_with_lowest_error,
                      data_set.classes, data_set.parameter_names,
                      animate = True, animation_rate = animation_rate,
                      x_axis_vector_index = x_axis_vector_index,
                      y_axis_vector_index = y_axis_vector_index)


# /////////////////////////////////////////////////////////////////////////// #
def kohonen_iteration(
        vectors: Tuple[Vector, ...],
        initial_neurons: Tuple[int, ...],
        classes,  # TODO
        parameter_names,  # TODO
        *,
        animate: bool = False,
        animation_rate: float = 0.001,
        x_axis_vector_index: int = 0,
        y_axis_vector_index: int = 1) \
        -> float:
    """ TODO
    """
    # ----------------------------------------------------------------------- #
    # ....................................... Additional variables and settings
    # Main loop control
    should_iterate: bool = True
    clusters = []

    # Create and set up plot
    if animate:
        # Define keyboard support while plotting the graph
        def on_key_down(event):
            # Close current figure
            if event.key == 'c':
                nonlocal should_iterate
                should_iterate = False
                pyplot.close(pyplot.gcf())
            # Save plot to file
            elif event.key == 's':
                pyplot.savefig('plot_kohonen.png')

        hide_matplotlib_toolbar()
        set_matplotlib_fontsize(16)
        fig, ax = pyplot.subplots()
        toggle_matplotlib_fullscreen()
        fig.canvas.mpl_connect('key_press_event', on_key_down)

    # Pick initial neurons ............................................. STEP 0
    neurons: List[Vector] = [vectors[i] for i in initial_neurons]

    # Select initial constants ......................................... STEP 0
    maximum_iterations: int = 64
    initial_radius: float = euclidean_distance(numpy.min(vectors, axis = 0),
                                               numpy.max(vectors, axis = 0))
    initial_rate: float = 0.05

    # Perform a given number of iterations
    for current_iteration in range(maximum_iterations):
        if should_iterate:

            # Calculate proper parameters for this iteration
            neighbourhood_radius: float \
                = calculate_neighbourhood_radius(current_iteration,
                                                 maximum_iterations,
                                                 initial_radius)
            learning_rate: float \
                = calculate_learning_rate(current_iteration,
                                          maximum_iterations,
                                          initial_rate)

            best_matching_units: List[int] = []

            # Randomly select an input vector .......................... STEP 1
            for current_vector in get_random_shuffled_range(len(vectors)):

                # For given vector, select Best Matching Unit .......... STEP 2
                best_matching_unit: int \
                    = find_best_matching_unit(vectors[current_vector],
                                              neurons)
                best_matching_units.append(best_matching_unit)

                # Get neurons within neighbourhood of BMU .............. STEP 3
                bmu_neighbourhood: Tuple[int, ...] \
                    = get_best_matching_unit_neighbourhood(
                        neurons[best_matching_unit],
                        neighbourhood_radius,
                        neurons)

                # Update neuron weights within the neighbourhood ....... STEP 4
                for i in bmu_neighbourhood:
                    neurons[i] = neurons[i] \
                                 + learning_rate * calculate_gaussian_decay(
                            neighbourhood_radius,
                            neurons[best_matching_unit],
                            neurons[i]) \
                                 * (vectors[current_vector] - neurons[i])

            # Check for dead neurons ................................... STEP 5
            for i in range(len(neurons)):
                unique, counts = numpy.unique(neurons, axis = 0,
                                              return_counts = True)
                if len(unique) < len(neurons):
                    if numpy.array_equal(neurons[i], unique[numpy.where(
                            counts == numpy.max(counts))[0][0]]):
                        neurons[i] = vectors[random.randint(0, len(neurons))]

                # for u in unique:
                #     if euclidean_distance(neurons[i], u) < 0.0001:
                #         neurons[i] = vectors[random.randint(0, len(neurons))]

                if i not in best_matching_units:
                    neurons[i] = vectors[random.randint(0, len(neurons))]

            clusters = assign_data_to_nearest_clusters(vectors, neurons)

            if animate:
                draw_kohonen(vectors, classes,
                             parameter_names,
                             x_axis_vector_index, y_axis_vector_index,
                             len(neurons),
                             neurons, clusters,
                             current_iteration)
                pyplot.pause(animation_rate)

        # centroids = calculate_centers_of_clusters(data, k, clusters)
        # for centroid in previous_centroids:
        #     if numpy.array_equal(centroid, centroids):
        #         should_iterate = False
        #         break
        # previous_centroids.append(centroids)

    if animate:
        pyplot.show()

    #Compute quantisation error for all clusters
    cluster_errors: List[float] = []

    for n in range(len(neurons)):
        vectors_in_nth_cluster = []
        for x in range(len(clusters)):
            if clusters[x] == n:
                vectors_in_nth_cluster.append(vectors[x])

        cluster_errors.append(numpy.std(vector_from_list(
                vectors_in_nth_cluster)))

    return sum(cluster_errors)

def get_best_matching_unit_neighbourhood(
        best_matching_unit: Vector,
        radius: float,
        neurons: Tuple[Vector, ...]) \
        -> Tuple[int, ...]:
    neighbourhood: List[int] = []
    for i in range(len(neurons)):
        if euclidean_distance(best_matching_unit, neurons[i]) <= radius:
            neighbourhood.append(i)
    return tuple(neighbourhood)


def calculate_neighbourhood_radius(
        current_iteration: int,
        maximum_iterations: int,
        initial_radius: float) \
        -> float:
    return sigma(initial_radius, current_iteration,
                 maximum_iterations / math.log(initial_radius))


def calculate_learning_rate(
        current_iteration: int,
        maximum_iterations: int,
        initial_rate: float) \
        -> float:
    return sigma(initial_rate, current_iteration, maximum_iterations)


def calculate_gaussian_decay(
        radius: float,
        vector: Vector,
        neuron: Vector) \
        -> float:
    # distance = euclidean_distance(neuron, vector)
    # return numpy.e**(-((distance**2) / (2 * radius**2)))
    if numpy.array_equal(vector, neuron):
        return 1
    else:
        return 0

def sigma(
        o: float,
        t: int,
        l: float) \
        -> float:
    return o * numpy.e**(-(t / l))


def find_best_matching_unit(
        vector: Vector,
        neurons: Tuple[Vector, ...]) \
        -> int:
    distances: Tuple[float, ...] = tuple(
            [euclidean_distance(neuron, vector)
             for neuron in neurons])
    return distances.index(min(distances))


def get_random_shuffled_range(
        n: int) \
        -> List[int]:
    return random.sample(range(n), k = n)


# /////////////////////////////////////////////////////////////////////////// #
def get_vectors_from_data_set(
        data_set: ClusteringData,
        mode: DataMode = DataMode.DEFAULT) \
        -> Tuple[Vector]:
    """ Select appropriate mode of data from data set
    """
    # ----------------------------------------------------------------------- #
    if mode == DataMode.NORMALISED:
        return data_set.data_normalised
    elif mode == DataMode.STANDARDISED:
        return data_set.data_standardised
    else:
        return data_set.data


# /////////////////////////////////////////////////////////////////////////// #
def get_column(
        two_dimensional_list: List[List],
        n: int) \
        -> List:
    """ Return n-th column of two dimensional list.

    Parameters
    ----------
    two_dimensional_list : List[List]
        Two dimensional list of which the column should be returned.
    n : int
        Number of column to return.

    Returns
    -------
    List
        N-th column of provided two dimensional list.

    """
    return [row[n] for row in two_dimensional_list]


# /////////////////////////////////////////////////////////////////////////// #
def pick_random_neurons(
        n: int,
        vectors: Tuple[Vector, ...]) \
        -> Tuple[int, ...]:
    """ Return indices of n random vectors
    """
    return tuple(get_random_shuffled_range(len(vectors))[0:n])


# /////////////////////////////////////////////////////////////////////////// #
def euclidean_distance(
        p: Vector,
        q: Vector) \
        -> float:
    """ Compute Euclidean distance between p and q vectors
    """
    return numpy.linalg.norm(p - q)


# /////////////////////////////////////////////////////////////////////////// #
def assign_data_to_nearest_clusters(
        data: Tuple[Vector, ...],
        centroids: Tuple[Vector, ...]) \
        -> Tuple[int]:
    clusters: List[int] = []

    for vector in data:
        lengths: Tuple[float] \
            = tuple([euclidean_distance(vector, centroid)
                     for centroid in centroids])

        clusters.append(lengths.index(min(lengths)))

    return tuple(clusters)


# /////////////////////////////////////////////////////////////////////////// #
def draw_kohonen(data, classes, names, n, m, k, centroids, clusters,
                 iteration_number):
    # Get and clear current axes
    ax = pyplot.gca()
    ax.clear()

    clustered_data = [[] for i in range(k)]

    for i in range(len(data)):
        clustered_data[clusters[i]].append([data[i][n], data[i][m]])

    color_map = pyplot.get_cmap('rainbow')(numpy.linspace(0, 1, k))
    markers: List[str] = ['o', 's', 'p', 'P', '*', 'D']

    x = [[[] for j in range(len(set(classes)))] for i in range(k)]
    y = [[[] for j in range(len(set(classes)))] for i in range(k)]
    stats = [[0 for j in range(len(set(classes)))] for i in range(k)]
    for i, vector in enumerate(data):
        for j, cl in enumerate(set(classes)):
            if classes[i] == cl:
                x[clusters[i]][j].append(vector[n])
                y[clusters[i]][j].append(vector[m])
                stats[clusters[i]][j] += 1

    for i in range(k):
        for j, cl in enumerate(set(classes)):
            ax.plot(x[i][j], y[i][j], markers[j % 6], color = color_map[i],
                    markersize = 6)

    winning_class = [stats[i].index(max(stats[i])) for i in range(k)]
    result = round(
            statistics.mean([stats[i][winning_class[i]] / classes.count(
                    list(set(classes))[winning_class[i]]) for i in
                             range(k)]) * 100, 2)

    for i in range(len(centroids)):
        ax.plot(centroids[i][n],
                centroids[i][m],
                '^', color = color_map[i],
                markersize = 18, markeredgewidth = 1.5,
                markeredgecolor = 'k')

    # Set axis parameters
    ax.set(title = 'Zadanie 1 - Wykresy (Metoda Kohonena)\nIteracja: '
                   + str(iteration_number) + '\nSkuteczność: ' + str(
            result)
                   + '%',
           xlabel = names[n],
           ylabel = names[m])


# /////////////////////////////////////////////////////////////////////////// #
def calculate_centers_of_clusters(data, k, clusters) \
        -> Tuple[Vector]:
    centers: List[Vector] = []

    clustered_data = [[] for i in range(k)]

    for i in range(len(data)):
        clustered_data[clusters[i]].append(data[i])

    for i in range(k):
        if clustered_data[i]:
            centers.append(numpy.mean(clustered_data[i], axis = 0))
        else:
            centers.append(pick_random_centroids(data, k)[0])

    return tuple(centers)


# /////////////////////////////////////////////////////////////////////////// #
def print_status_bar(
        title: str,
        a: int,
        b: int,
        n: int = 10) \
        -> None:
    print('\r', title, ': |', '=' * int((a / b) * n),
          '-' if a != b else '',
          ' ' * (n - int((a / b) * n) - 1),
          '| (', a, ' / ', b, ')',
          sep = '', end = '', flush = True)


# /////////////////////////////////////////////////////////////////////////// #
def toggle_matplotlib_fullscreen():
    pyplot.get_current_fig_manager().full_screen_toggle()


# /////////////////////////////////////////////////////////////////////////// #
def hide_matplotlib_toolbar():
    pyplot.rcParams['toolbar'] = 'None'


# /////////////////////////////////////////////////////////////////////////// #
def set_matplotlib_fontsize(
        size: int) \
        -> None:
    pyplot.rcParams.update({'font.size': size})

# /////////////////////////////////////////////////////////////////////////// #
