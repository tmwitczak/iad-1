# /////////////////////////////////////////////////////////////////// Imports #
import csv
import math
import random
import numpy

import matplotlib
from matplotlib import pyplot


# /////////////////////////////////////////////////////////////////////////// #
def load_csv_file_into_array(filename):
    result = []

    with open(filename) as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            result.append(tuple(row))

    return tuple(result)


def get_column(two_dimensional_array, n):
    column = []

    for row in two_dimensional_array:
        column.append(float(row[n]))

    return column


def create_two_dimensional_array(n, m):
    return [[None for j in range(m)] for i in range(n)]


def define_centroids(data, k):
    centroids = []

    for i in range(k):
        centroid = []
        for j in range(len(data[0])):
            centroid.append(random.uniform(
                    float(min(get_column(data, j))),
                    float(max(get_column(data, j)))))
        centroids.append(tuple(centroid))

    return centroids


# def vector_difference(a, b):
# result = []


def euclidean_distance(p, q):
    return math.sqrt(sum([math.pow(float(p[i]) - float(q[i]), 2) for i in
                          range(len(p))]))


def assign_clusters(data, centroids):
    clusters = []

    for i in range(len(data)):
        lengths = []
        for j in range(len(centroids)):
            lengths.append(euclidean_distance(data[i], centroids[j]))
        clusters.append(lengths.index(min(lengths)))

    return clusters


def toggle_matplotlib_fullscreen():
    pyplot.get_current_fig_manager().full_screen_toggle()


def hide_matplotlib_toolbar():
    matplotlib.rcParams["toolbar"] = "None"


def draw_k_means(data, n, m, k, centroids, clusters, iteration_number):
    # Get and clear current axes
    ax = pyplot.gca()
    pyplot.cla()

    # Set axis parameters
    ax.set(title = "Zadanie 1 - Wykresy (Metoda K-Średnich)\nIteracja: "
                   + str(iteration_number))


    clustered_data = [[] for i in range(k)]

    for i in range(len(data)):
        clustered_data[clusters[i]].append([data[i][n], data[i][m]])

    colors = "rgbcmykw"

    for i in range(len(clustered_data)):
        ax.plot(get_column(clustered_data[i], 0),
                get_column(clustered_data[i], 1),
                colors[i] + "o")

    ax.plot(get_column(centroids, n),
            get_column(centroids, m),
            "yo", markersize = 12)

    # Plot the graph
    pyplot.show()


def move_centroids(data, k, centroids, clusters):
    new_centers = []

    clustered_data = [[] for i in range(k)]

    for i in range(len(data)):
        clustered_data[clusters[i]].append([data[i][x] for x in range(len(
                data[0]))])

    for i in range(len(clustered_data)):
        center = numpy.array([0 for x in range(len(clustered_data[0][0]))],
                             float)
        for j in range(len(clustered_data[i])):
            center += numpy.array(clustered_data[i][j], float)

        center /= len(clustered_data[i])
        new_centers.append(center)

    return new_centers


# /////////////////////////////////////////////////////////////////////////// #
def k_means(data, k, i, j):
    iteration_number = 1

    # For given k, pick k initial centroids and assign clusters
    centroids = define_centroids(data, k)
    clusters = assign_clusters(data, centroids)

    # Perform next iteration of the algorithm
    def next_iteration():
        nonlocal centroids, clusters, iteration_number
        centroids = move_centroids(data, k, centroids, clusters)
        clusters = assign_clusters(data, centroids)
        iteration_number = iteration_number + 1
        draw_k_means(data, i, j, k, centroids, clusters, iteration_number)

    # Define keyboard support for the plot
    def on_key_down(event):
        # Draw next plot
        if event.key == 'x':
            next_iteration()

        # Reset 
        if event.key == 'r':
            nonlocal centroids, clusters, iteration_number
            centroids = define_centroids(data, k)
            clusters = assign_clusters(data, centroids)
            iteration_number = 1
            draw_k_means(data, i, j, k, centroids, clusters, iteration_number)

        # Close current figure
        if event.key == 'c':
            pyplot.close(pyplot.gcf())

    # Create and draw plot
    hide_matplotlib_toolbar()
    fig, ax = pyplot.subplots()
    fig.canvas.mpl_connect("key_press_event", on_key_down)
    toggle_matplotlib_fullscreen()
    draw_k_means(data, i, j, k, centroids, clusters, iteration_number)


# ////////////////////////////////////////////////////////////////////// Main #
def main():
    def mean(data):
        return sum(data) / len(data)

    def std(data):
        srednia = mean(data)
        suma = 0
        for i in range(len(data)):
            suma += math.pow(data[i] - srednia, 2)
        suma /= (len(data) - 1)
        return math.sqrt(suma)

    iris_data = load_csv_file_into_array("data\_iris.data")
    k = 3

    iris_2 = []
    for i in range(len(iris_data)):
        iris_2.append(iris_data[i][0:len(iris_data[0]) - 1])

    iris_3 = []
    for i in range(len(iris_2)):
        iris = []
        for j in range(len(iris_2[0])):
            iris.append((float(iris_2[i][j]) - mean(get_column(iris_2,
                                                               j))) / std(
                    get_column(iris_2, j)))
        iris_3.append(iris)

    k_means(iris_3, k, 2, 3)

    wine_dataa = load_csv_file_into_array("data\_wine.data")

    wine_data = []
    for i in range(len(wine_dataa)):
        wine_data.append(wine_dataa[i][1:len(wine_dataa[0])])

    wine_2 = []
    for i in range(len(wine_data)):
        wine = []
        for j in range(len(wine_data[0])):
            wine.append((float(wine_data[i][j]) - mean(get_column(wine_data,
                                                               j))) / std(
                    get_column(wine_data, j)))
        wine_2.append(wine)

    k_means(wine_2, k, 6, 7)

    # # inputing parameters from user
    # kValue = input("Podaj wartosc K: ")
    # tolerance = input("Podaj wartosc tolerancji: ")
    # maxNumOfIter = input("Podaj maksymalna liczbe iteracji: ")
    # kMeans = KMeansAlgorithm(kValue, tolerance, maxNumOfIter)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == "__main__":
    main()

# /////////////////////////////////////////////////////////////////////////// #
