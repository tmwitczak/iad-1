# /////////////////////////////////////////////////////////////////// Imports #
from image_compression_algorithm import compress_image


# ////////////////////////////////////////////////////////////////////// Main #
def main() \
        -> None:
    """ Compress images using k-means clustering
    """
    # ----------------------------------------------------------------------- #
    # Print program's title
    print('[ Kompresja obrazu ]')

    # Input image parameters
    input_filename: str \
        = input('Podaj ścieżkę do obrazu wejściowego: ')

    output_filename: str \
        = input('Podaj ścieżkę do obrazu wyjściowego: ')

    frame_size: int \
        = int(input('Podaj rozmiar ramki: '))

    number_of_clusters: int \
        = int(input('Podaj liczbę klastrów: '))

    number_of_clusterisations: int \
        = int(input('Podaj liczbę powtórzeń klasteryzacji: '))

    number_of_maximum_iterations: int \
        = int(input('Podaj maksymalną liczbę iteracji: '))

    # Compress given image
    compress_image(input_filename,
                   output_filename,
                   frame_size,
                   number_of_clusters,
                   number_of_clusterisations,
                   number_of_maximum_iterations)


# /////////////////////////////////////////////////////////// Execute program #
if __name__ == '__main__':
    main()

# /////////////////////////////////////////////////////////////////////////// #
