# /////////////////////////////////////////////////////////////////// Imports #
from src.file_reading import DataSets, load_data_sets, ClusteringData
from src.k_means_algorithm import k_means, KMeansDataMode


# ////////////////////////////////////////////////////////////////////// Main #
from src.kohonen_network_algorithm import KohonenDataMode, kohonen


def main() -> None:
    """ Analyse data sets with different clustering algorithms
    """
    # ----------------------------------------------------------------------- #
    data_sets: DataSets = load_data_sets()

    data_set: ClusteringData = data_sets.wine
    number_of_clusters: int = 3
    iterations: int = 2
    x_axis_vector_index: int = 0
    y_axis_vector_index: int = 1
    animation_rate: float = 0.0001
    mode: KohonenDataMode = KohonenDataMode.STANDARDISED

    kohonen(data_set,
            number_of_clusters,
            iterations = iterations,
            i = x_axis_vector_index,
            j = y_axis_vector_index,
            mode = mode,
            animation_rate = animation_rate)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == '__main__':
    main()

# /////////////////////////////////////////////////////////////////////////// #
