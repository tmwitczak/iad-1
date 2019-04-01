# /////////////////////////////////////////////////////////////////// Imports #
from src.file_reading import DataSets, load_data_sets, ClusteringData, DataMode
from src.k_means_algorithm import k_means, DataMode


# ////////////////////////////////////////////////////////////////////// Main #
from src.kohonen_network_algorithm import DataMode, kohonen
from src.NeuralGasAlgorithm import neural_gas


def main() -> None:
    """ Analyse data sets with different clustering algorithms
    """
    # ----------------------------------------------------------------------- #
    data_sets: DataSets = load_data_sets()

    data_set: ClusteringData = data_sets.iris
    number_of_neurons: int = 20
    number_of_iterations: int = 1
    x_axis_vector_index: int = 2
    y_axis_vector_index: int = 3
    animation_rate: float = 0.000001
    mode: DataMode = DataMode.STANDARDISED

    kohonen(data_set,
            number_of_neurons,
            iterations = number_of_iterations,
            x_axis_vector_index = x_axis_vector_index,
            y_axis_vector_index = y_axis_vector_index,
            mode = mode,
            animation_rate = animation_rate)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == '__main__':
    main()

# /////////////////////////////////////////////////////////////////////////// #
