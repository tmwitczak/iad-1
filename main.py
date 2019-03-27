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


def draw_k_means(data, n, m, k, centroids, clusters):
    # Set main parameters
    # hide_matplotlib_toolbar()

    # Create main plot
    # fig, ax = pyplot.subplots()
    # ax.set(title="Zadanie 1 - Wykresy (Metoda K-Średnich)")

    # fig.canvas.mpl_connect("key_press_event", onkeydown)

    # pyplot.suptitle("Zadanie 1 - Wykresy (K-Means)")

    # Prepare the first graph (sepal length x sepal width)
    # pyplot.subplot(1, 2, 1)

    ax = pyplot.gca()
    pyplot.cla()

    clustered_data = [[] for i in range(k)]

    for i in range(len(data)):
        clustered_data[clusters[i]].append([data[i][n], data[i][m]])

    ax.plot(get_column(centroids, n),
            get_column(centroids, m),
            "yo", markersize = 12)
    ax.plot(get_column(clustered_data[0], 0),
            get_column(clustered_data[0], 1),
            "ro")
    ax.plot(get_column(clustered_data[1], 0),
            get_column(clustered_data[1], 1),
            "go")
    ax.plot(get_column(clustered_data[2], 0),
            get_column(clustered_data[2], 1),
            "bo")
    # pyplot.plot(get_column(centroids, 0),
    #           get_column(centroids, 1),
    #          "r+")

    """pyplot.plot(get_column(data, 0),
                get_column(data, 1),
                "r+",
                label = "Iris setosa")
    pyplot.plot(iris_versicolor.iloc[:, 0],
                iris_versicolor.iloc[:, 1],
                "g+",
                label = "Iris versicolor")
    pyplot.plot(iris_virginica.iloc[:, 0],
                iris_virginica.iloc[:, 1],
                "b+",
                label = "Iris virginica")
    pyplot.title("Zależność szerokości od długości działki kielicha")
    pyplot.xlabel("Długość działki kielicha [cm]")
    pyplot.ylabel("Szerokość działki kielicha [cm]")
    pyplot.legend()

    # Prepare the second graph (petal length x petal width)
    pyplot.subplot(1, 2, 2)
    pyplot.plot(iris_setosa.iloc[:, 2],
                iris_setosa.iloc[:, 3],
                "r+",
                label = "Iris setosa")
    pyplot.plot(iris_versicolor.iloc[:, 2],
                iris_versicolor.iloc[:, 3],
                "g+",
                label = "Iris versicolor")
    pyplot.plot(iris_virginica.iloc[:, 2],
                iris_virginica.iloc[:, 3],
                "b+",
                label = "Iris virginica")
    pyplot.title("Zależność szerokości od długości płatka")
    pyplot.xlabel("Długość płatka [cm]")
    pyplot.ylabel("Szerokość płatka [cm]")
    pyplot.legend()
"""
    # Plot final graph
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


def k_means(data, k):
    """
    :param data:
    :param k:
    :return:
    """
    centroids = define_centroids(data, k)
    clusters = assign_clusters(data, centroids)

    def next_iteration():
        nonlocal centroids
        nonlocal clusters
        centroids = move_centroids(data, k, centroids, clusters)
        clusters = assign_clusters(data, centroids)
        draw_k_means(data, 2, 3, k, centroids, clusters)

    def on_key_down(event):
        # Draw next plot
        if event.key == 'x':
            next_iteration()

        # Reset 
        if event.key == 'r':
            nonlocal centroids
            nonlocal clusters
            centroids = define_centroids(data, k)
            clusters = assign_clusters(data, centroids)
            draw_k_means(data, 2, 3, k, centroids, clusters)

        # Close current figure
        if event.key == 'c':
            pyplot.close(pyplot.gcf())

    # Set main parameters
    hide_matplotlib_toolbar()

    # Create main plot
    fig, ax = pyplot.subplots()
    # ax.set(title = "Zadanie 1 - Wykresy (Metoda K-Średnich)")

    fig.canvas.mpl_connect("key_press_event", on_key_down)
    toggle_matplotlib_fullscreen()

    draw_k_means(data, 2, 3, k, centroids, clusters)


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

    k_means(iris_3, k)

    # # inputing parameters from user
    # kValue = input("Podaj wartosc K: ")
    # tolerance = input("Podaj wartosc tolerancji: ")
    # maxNumOfIter = input("Podaj maksymalna liczbe iteracji: ")
    # kMeans = KMeansAlgorithm(kValue, tolerance, maxNumOfIter)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == "__main__":
    main()

# /////////////////////////////////////////////////////////////////////////// #
