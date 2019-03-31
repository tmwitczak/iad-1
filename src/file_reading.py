# /////////////////////////////////////////////////////////////////// Imports #
import csv
import statistics
from typing import Any, List, NamedTuple, Tuple

import numpy

# ////////////////////////////////////////////////////////////////// Typedefs #
Vector = numpy.ndarray


def vector_from_list(
        x: List[float]) \
        -> Vector:
    return numpy.array(x)


def empty_vector(
        length: int) \
        -> Vector:
    return numpy.empty(shape = length)


# /////////////////////////////////////////////////////////////////// Classes #
class ClusteringData(NamedTuple):
    data: Tuple[Vector, ...]
    data_normalised: Tuple[Vector, ...]
    data_standardised: Tuple[Vector, ...]
    classes: Tuple[str, ...]
    parameter_names: Tuple[str, ...]


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
    iris_parameter_names: Tuple[str, ...] = ('Sepal length',
                                             'Sepal width',
                                             'Petal length',
                                             'Petal width')
    iris: ClusteringData = load_clustering_data_from_csv_file(
            iris_csv_filename, iris_class_column_number,
            iris_parameter_names)

    wine_csv_filename: str = 'data/wine.data'
    wine_class_column_number: int = 0
    wine_parameter_names: Tuple[str, ...] = ('Malic acid',
                                             'Ash',
                                             'Alcalinity of ash',
                                             'Magnesium',
                                             'Total phenols',
                                             'Flavanoids',
                                             'Nonflavanoid phenols',
                                             'Proanthocyanins',
                                             'Color intensity',
                                             'Hue',
                                             'OD280/OD315 of diluted wines',
                                             'Proline')
    wine: ClusteringData = load_clustering_data_from_csv_file(
            wine_csv_filename, wine_class_column_number,
            wine_parameter_names)

    abalone_csv_filename: str = 'data/abalone.data'
    abalone_class_column_number: int = 0
    abalone_parameter_names: Tuple[str, ...] = ('Length',
                                                'Diameter',
                                                'Height',
                                                'Whole weight',
                                                'Shucked weight',
                                                'Viscera weight',
                                                'Shell weight',
                                                'Rings')
    # 'Rings')
    abalone: ClusteringData = load_clustering_data_from_csv_file(
            abalone_csv_filename, abalone_class_column_number,
            abalone_parameter_names)

    # Return data sets
    return DataSets(iris, wine, abalone)


# /////////////////////////////////////////////////////////////////////////// #
def load_clustering_data_from_csv_file(
        csv_filename: str,
        classes_column_number: int,
        parameter_names: Tuple[str, ...]) \
        -> ClusteringData:
    """ Load data from csv file and separate numeric values from classes.

    Parameters
    ----------
    csv_filename : str
        Path to csv file.
    classes_column_number : int
        Number of column containing class identifier.
    parameter_names : Tuple[str, ...]
        Names of subsequent parameters in data vector

    Returns
    -------
    ClusteringData
        Set of data for further clustering.

    """
    # ----------------------------------------------------------------------- #
    # Clustering data attributes
    data: List[Vector] = []
    data_normalised: List[Vector] = []
    data_standardised: List[Vector] = []
    classes: List[str] = []

    # Load data and classes
    with open(csv_filename) as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            data.append(
                    vector_from_list(
                            [float(x) for x in row[0:classes_column_number]]
                            + [float(x) for x in row[(classes_column_number
                                                      + 1):len(row)]]))
            classes.append(row[classes_column_number])

    # Normalise and standardise data
    min_vector: Vector = empty_vector(len(data[0]))
    max_vector: Vector = empty_vector(len(data[0]))
    mean_vector: Vector = empty_vector(len(data[0]))
    stdev_vector: Vector = empty_vector(len(data[0]))

    for i in range(len(data[0])):
        min_vector[i] = min(get_column(data, i))
        max_vector[i] = max(get_column(data, i))
        mean_vector[i] = statistics.mean(get_column(data, i))
        stdev_vector[i] = statistics.stdev(get_column(data, i))

    for vector in data:
        data_normalised.append(
                (vector - min_vector) / (max_vector - min_vector))
        data_standardised.append((vector - mean_vector) / stdev_vector)

    # Return whole data set
    return ClusteringData(tuple(data), tuple(data_normalised),
                          tuple(data_standardised), tuple(classes),
                          tuple(parameter_names))


# /////////////////////////////////////////////////////////////////////////// #
def get_column(
        two_dimensional_list: List[List[Any]],
        n: int) \
        -> List:
    """ Return n-th column of two dimensional list.

    Parameters
    ----------
    two_dimensional_list : List[List[Any]]
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
