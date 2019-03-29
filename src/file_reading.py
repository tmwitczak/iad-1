# /////////////////////////////////////////////////////////////////// Imports #
import csv
from typing import List, NamedTuple, Tuple
import statistics

import numpy

# ////////////////////////////////////////////////////////////////// Typedefs #
Vector = numpy.array


class ClusteringData(NamedTuple):
    data: Tuple[Vector]
    data_normalised: Tuple[Vector]
    data_standardised: Tuple[Vector]
    classes: Tuple[str]


class DataSets(NamedTuple):
    iris: ClusteringData
    wine: ClusteringData
    abalone: ClusteringData


# /////////////////////////////////////////////////////////////////////////// #
def load_data_sets() \
        -> DataSets:
    """ Load pre-chosen data sets from csv file for further clustering.

    Returns
    -------
    DataSets
        Named tuple for separate data sets: iris, wine, abalone.

    """
    # ----------------------------------------------------------------------- #
    # Load data sets from csv files
    iris_csv_filename: str = 'data/iris.data'
    iris_class_column_number: int = 4
    iris: ClusteringData = load_clustering_data_from_csv_file(
            iris_csv_filename, iris_class_column_number)

    wine_csv_filename: str = 'data/wine.data'
    wine_class_column_number: int = 0
    wine: ClusteringData = load_clustering_data_from_csv_file(
            wine_csv_filename, wine_class_column_number)

    abalone_csv_filename: str = 'data/abalone.data'
    abalone_class_column_number: int = 0
    abalone: ClusteringData = load_clustering_data_from_csv_file(
            abalone_csv_filename, abalone_class_column_number)

    # Return data sets
    return DataSets(iris, wine, abalone)


# /////////////////////////////////////////////////////////////////////////// #
def load_clustering_data_from_csv_file(
        csv_filename: str,
        classes_column_number: int) \
        -> ClusteringData:
    """ Load data from csv file and separate numeric values from classes.

    Parameters
    ----------
    csv_filename : str
        Path to csv file.
    classes_column_number : int
        Number of column containing class identifier.

    Returns
    -------
    ClusteringData
        Set of data for further clustering.

    """
    # ----------------------------------------------------------------------- #
    data: List[Vector] = []
    data_normalised: List[Vector] = []
    data_standardised: List[Vector] = []
    classes: List[str] = []

    with open(csv_filename) as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            data.append(
                    Vector([float(x) for x in row[0:classes_column_number]]
                           + [float(x) for x in row[(classes_column_number + 1)
                                                    :len(row)]]))
            classes.append(row[classes_column_number])

    min_vector: Vector = numpy.empty(shape = len(data[0]))
    max_vector: Vector = numpy.empty(shape = len(data[0]))
    mean_vector: Vector = numpy.empty(shape = len(data[0]))
    stdev_vector: Vector = numpy.empty(shape = len(data[0]))

    for i in range(len(data[0])):
        min_vector[i] = min(get_column(data, i))
        max_vector[i] = max(get_column(data, i))
        mean_vector[i] = statistics.mean(get_column(data, i))
        stdev_vector[i] = statistics.stdev(get_column(data, i))

    for vector in data:
        data_normalised.append(
                (vector - min_vector) / (max_vector - min_vector))
        data_standardised.append((vector - mean_vector) / stdev_vector)

    return ClusteringData(tuple(data), tuple(data_normalised),
                          tuple(data_standardised), tuple(classes))


# /////////////////////////////////////////////////////////////////////////// #
def get_column(
        two_dimensional_list: List[List],
        n: int)\
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
