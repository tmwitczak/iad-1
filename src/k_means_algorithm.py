# /////////////////////////////////////////////////////////////////// Imports #
import random
import statistics
from enum import Enum
from typing import List, Tuple

import numpy
from matplotlib import pyplot

from src.file_reading import ClusteringData, Vector, vector_from_list, DataMode


# ///////////////////////////////////////////////////////// K-means algorithm #
def k_means(
        data_set: ClusteringData,
        k: int,
        *,
        iterations: int = 10,
        animation_rate: float = 0.001,
        x_axis_vector_index: int = 0,
        y_axis_vector_index: int = 1,
        mode: DataMode = DataMode.DEFAULT) \
        -> None:
    """ Find clusters using k-means algorithm.

    Parameters
    ----------
    animation_rate  #TODO
    data_set : ClusteringData
        Data set for finding clusters.
    k : int
        Number of clusters to find
    iterations
        Number of times the algorithm will be executed, the result with the
        lowest quantisation error will be chosen.
    x_axis_vector_index : int
        Index of vector element to plot on x axis.
    y_axis_vector_index : int
        Index of vector element to plot on y axis.
    mode : DataMode
        Select whether data is normalised or standardised

    """
    # ----------------------------------------------------------------------- #
    # Choose appropriate data mode
    data: Tuple[Vector] = get_data_from_data_set(data_set, mode)

    # Perform a given number of iterations of k-means algorithm
    # and choose the result with the lowest quantisation error
    initial_centroids: List[Tuple[Vector]] = []
    quantisation_errors: List[float] = []

    print('[ K-Means ]')
    for x in range(iterations):
        print_status_bar('> Iterations', x + 1, iterations)
        initial_centroids.append(
                pick_random_centroids(data, k))
        quantisation_errors.append(
                k_means_iteration(data_set, data, k, initial_centroids[x]))

    final_initial_centroid: Tuple[Vector] = initial_centroids[
        quantisation_errors.index(min(quantisation_errors))]

    # Draw animation for the best iteration
    k_means_iteration(data_set, data, k, final_initial_centroid, True,
                      animation_rate, x_axis_vector_index, y_axis_vector_index)


# /////////////////////////////////////////////////////////////////////////// #
def k_means_iteration(
        data_set: ClusteringData,
        data: Tuple[Vector],
        k: int,
        initial_centroids: Tuple[Vector],
        animate: bool = False,
        animation_rate: float = 0.001,
        i: int = 0,
        j: int = 1) \
        -> float:
    """ TODO
    """
    # ----------------------------------------------------------------------- #
    # For given number k, pick k initial centroids and assign clusters
    centroids: Tuple[Vector] = initial_centroids
    clusters: Tuple[int]
    should_iterate: bool = True

    # Define keyboard support while plotting the graph
    def on_key_down(event):
        # Close current figure
        if event.key == 'c':
            nonlocal should_iterate
            should_iterate = False
            pyplot.close(pyplot.gcf())
        # Save plot to file
        elif event.key == 's':
            pyplot.savefig('plot_k_means.png')

    # Create and draw plot
    if animate:
        hide_matplotlib_toolbar()
        set_matplotlib_fontsize(16)
        fig, ax = pyplot.subplots()
        toggle_matplotlib_fullscreen()
        fig.canvas.mpl_connect('key_press_event', on_key_down)

    # Assign every point to the nearest cluster and calculate new centroids
    previous_centroids: List[Tuple[Vector]] = []
    current_iteration: int = 1

    while should_iterate:
        clusters = assign_data_to_nearest_clusters(data, centroids)
        if animate:
            draw_k_means(data, data_set.classes, data_set.parameter_names,
                         i, j, k,
                         centroids, clusters,
                         current_iteration)
            pyplot.pause(animation_rate)
        centroids = calculate_centers_of_clusters(data, k, clusters)
        for centroid in previous_centroids:
            if numpy.array_equal(centroid, centroids):
                should_iterate = False
                break
        previous_centroids.append(centroids)

        current_iteration += 1

    if animate:
        pyplot.show()

    # Compute quantisation error for all clusters
    cluster_errors: List[float] = []

    for n in range(k):
        vectors_in_nth_cluster = []
        for x in range(len(clusters)):
            if clusters[x] == n:
                vectors_in_nth_cluster.append(data[x])

        cluster_errors.append(numpy.std(vector_from_list(
                vectors_in_nth_cluster)))

    return sum(cluster_errors)


# /////////////////////////////////////////////////////////////////////////// #
def get_data_from_data_set(
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
def pick_random_centroids(
        data: Tuple[Vector],
        k: int) \
        -> Tuple[Vector]:
    return tuple(
            [vector_from_list(
                    [random.uniform(
                            float(min(get_column(data, j))),
                            float(max(get_column(data, j))))
                        for j in range(len(data[0]))])
                for i in range(k)])


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
        data: ClusteringData,
        centroids: Tuple[Vector]) \
        -> Tuple[int]:
    clusters: List[int] = []

    for vector in data:
        lengths: Tuple[float] \
            = tuple([euclidean_distance(vector, centroid)
                     for centroid in centroids])

        clusters.append(lengths.index(min(lengths)))

    return tuple(clusters)


# /////////////////////////////////////////////////////////////////////////// #
def draw_k_means(data, classes, names, n, m, k, centroids, clusters,
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
    result = round(statistics.mean([stats[i][winning_class[i]] / classes.count(
            list(set(classes))[winning_class[i]]) for i in range(k)]) * 100, 2)

    for i in range(len(centroids)):
        ax.plot(centroids[i][n],
                centroids[i][m],
                '^', color = color_map[i],
                markersize = 18, markeredgewidth = 1.5,
                markeredgecolor = 'k')

    # Set axis parameters
    ax.set(title = 'Zadanie 1 - Wykresy (Metoda K-Średnich)\nIteracja: '
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
