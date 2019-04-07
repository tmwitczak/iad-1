# /////////////////////////////////////////////////////////////////// Imports #
import random
from typing import Callable, List, Tuple

import matplotlib.pyplot
import numpy
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, XYZColor
from matplotlib.image import imread
from numpy import argmin, array, ndarray, std, where, zeros
from numpy.linalg import norm
from numpy.random import randint


# /////////////////////////////////////////////// Image compression algorithm #
def compress_image(
        input_filename: str,
        output_filename: str,
        frame_size: int,
        number_of_clusters: int,
        number_of_clusterisations: int,
        number_of_maximum_iterations: int) \
        -> None:
    """ Compress given image using k-means clustering.
    """
    # ----------------------------------------------------------------------- #
    print(' > Ładowanie pliku obrazu:', input_filename)
    rgb_pixels: List[List[ndarray]] \
        = load_image_from_file(input_filename)

    print(' > Konwersja pikseli między przestrzeniami barw: sRGB -> XYZ')
    xyz_pixels: List[List[ndarray]] \
        = convert_pixels_from_rgb_to_xyz(rgb_pixels)

    print(' > Pakowanie pikseli w ramki o wymiarach:',
          frame_size, 'x', frame_size)
    clustering_frames: List[ndarray] \
        = pack_pixels_into_frames(xyz_pixels, frame_size)

    print(' > Klasteryzacja ramek:')
    compressed_frames: List[ndarray] \
        = clusterise_frames(clustering_frames,
                            number_of_clusters,
                            number_of_clusterisations,
                            number_of_maximum_iterations)

    print(' > Rozpakowywanie ramek w poszczególne piksele')
    xyz_pixels_compressed: List[List[ndarray]] \
        = unpack_frames_into_pixels(compressed_frames, frame_size)

    print(' > Konwersja pikseli między przestrzeniami barw: XYZ -> sRGB')
    rgb_pixels_compressed: List[List[ndarray]] \
        = convert_pixels_from_xyz_to_rgb(xyz_pixels_compressed)

    print(' > Zapisywanie pliku obrazu:', output_filename)
    save_image_to_file(rgb_pixels_compressed, output_filename)


# ////////////////////////////////////////////////////////// compress_image < #
def load_image_from_file(
        filename: str) \
        -> List[List[ndarray]]:
    """ Load image from file using matplotlib.
    """
    # ----------------------------------------------------------------------- #
    return [[array(pixel)
             for pixel in row]
            for row in imread(filename)]


def convert_pixels_from_rgb_to_xyz(
        rgb_pixels: List[List[ndarray]]) \
        -> List[List[ndarray]]:
    """ Convert a two dimensional grid of pixels from RGB to XYZ color space
    """
    # ----------------------------------------------------------------------- #
    return convert_pixels_color_space(rgb_pixels, convert_rgb_to_xyz)


def pack_pixels_into_frames(
        pixels: List[List[ndarray]],
        frame_size: int) \
        -> List[ndarray]:
    """ Pack pixels into frames in form of vector.
    """
    # ----------------------------------------------------------------------- #
    frame_number: int = int((len(pixels) * len(pixels[0])) / (frame_size ** 2))

    frames: List[List[float]] = [[] for _ in range(frame_number)]

    for i, row in enumerate(pixels):
        for j, pixel in enumerate(row):
            for k, value in enumerate(pixel):
                frame_i: int = int(i / frame_size)
                frame_j: int = int(j / frame_size)
                frame_n: int = int(len(pixels[0]) / frame_size)
                frames[frame_i * frame_n + frame_j].append(value)

    return [array(frame) for frame in frames]


def clusterise_frames(
        clustering_frames: List[ndarray],
        number_of_clusters: int,
        number_of_clusterisations: int,
        number_of_maximum_iterations: int) \
        -> List[ndarray]:
    """ Clusterise frames and return appropriate centroids.
    """
    # ----------------------------------------------------------------------- #
    # Perform k-means for a given number of times and
    # choose the result with the lowest quantisation error
    clusters: List[List[ndarray]] = []
    quantisation_errors: List[float] = []

    for i in range(number_of_clusterisations):
        print_status_bar('  - Postęp klasteryzacji',
                         i,
                         number_of_clusterisations)

        k_means_result: Tuple[List[ndarray], float] \
            = k_means(clustering_frames,
                      number_of_clusters,
                      number_of_maximum_iterations)
        clusters.append(k_means_result[0])
        quantisation_errors.append(k_means_result[1])

    print_status_bar('  - Postęp klasteryzacji', number_of_clusterisations,
                     number_of_clusterisations)
    print()
    return clusters[quantisation_errors.index(min(quantisation_errors))]


def unpack_frames_into_pixels(
        frames: List[ndarray],
        frame_size: int) \
        -> List[List[ndarray]]:
    """ Convert frames back to a two dimensional grid of pixels.
    """
    # ----------------------------------------------------------------------- #
    image_size: int = 512
    pixels: List[List[ndarray]] = [[zeros(3) for _ in range(image_size)]
                                   for _ in range(image_size)]
    unpacked_frames: List[ndarray] = [*frames]

    for i, row in enumerate(pixels):
        for j, pixel in enumerate(row):
            for k, value in enumerate(pixel):
                frame_i: int = int(i / frame_size)
                frame_j: int = int(j / frame_size)
                frame_n: int = int(len(pixels[0]) / frame_size)
                pixels[i][j][k] \
                    = unpacked_frames[frame_i * frame_n + frame_j][0]
                unpacked_frames[frame_i * frame_n + frame_j] \
                    = numpy.delete(
                        unpacked_frames[frame_i * frame_n + frame_j],
                        0)

    return pixels


def convert_pixels_from_xyz_to_rgb(
        rgb_pixels: List[List[ndarray]]) \
        -> List[List[ndarray]]:
    """ Convert a two dimensional grid of pixels from XYZ to RGB color space.
    """
    # ----------------------------------------------------------------------- #
    return convert_pixels_color_space(rgb_pixels, convert_xyz_to_rgb)


def save_image_to_file(
        pixels: List[List[ndarray]],
        filename: str) \
        -> None:
    """ Save pixels to file.
    """
    # ----------------------------------------------------------------------- #
    matplotlib.pyplot.imsave(filename, pixels)


# /////////////////////////////////////////////////////// clusterise_frames < #
def k_means(
        vectors: List[ndarray],
        number_of_clusters: int,
        number_of_maximum_iterations: int) \
        -> Tuple[List[ndarray], float]:
    """ Clusterise data and replace every vector with its centroid.
    """
    # ----------------------------------------------------------------------- #
    # For given number of clusters, pick initial centroids out of vectors
    centroids: List[ndarray] \
        = [vectors[i]
           for i in
           get_random_shuffled_range(len(vectors))[0:number_of_clusters]]
    cluster_indices: ndarray = zeros(0)
    clustered_vectors: List[ndarray] = []

    # Assign every point to the nearest cluster and calculate new centroids
    previous_centroids: List[List[ndarray]] = []
    current_iteration: int = 0
    len_vectors: int = len(vectors)

    should_iterate: bool = True
    while should_iterate:
        # Iteration index
        current_iteration += 1

        # Print information about current iteration
        print(' [Iteracja: ', current_iteration, ']',
              sep = '', end = '', flush = True)

        # Give every vector its cluster index
        cluster_indices = numpy.array([argmin([norm(vector - centroid)
                                               for centroid in centroids])
                                       for vector in vectors])

        # Get all vectors of a given clusters
        clustered_vectors \
            = [numpy.array([vectors[element_index]
                            for element_index in where(cluster_indices ==
                                                       cluster_index)[0]])
               for cluster_index in range(number_of_clusters)]

        # Calculate centroids
        centroids = [numpy.mean(cluster, axis = 0)
                     if cluster.any()
                     else vectors[randint(len_vectors)]
                     for cluster in clustered_vectors]

        # Check for ending condition
        # - reached maximum number of iterations
        # - centroids don't move or cycle
        if any([numpy.array_equal(centroid, centroids)
                for centroid in previous_centroids]) \
                or current_iteration == number_of_maximum_iterations:
            should_iterate = False
        else:
            previous_centroids.append(centroids)

        # Move console caret back for another iteration
        print(len(' [Iteracja: ' + str(current_iteration) + ']') * '\b',
              sep = '', end = '', flush = True)

    # Compute quantisation error for all clusters
    quantisation_error: float \
        = sum(std(cluster) for cluster in clustered_vectors)

    # Return clustered frames and quantisation error
    return ([centroids[cluster_index]
             for cluster_index in cluster_indices],
            quantisation_error)


def print_status_bar(
        title: str,
        a: int,
        b: int,
        n: int = 10) \
        -> None:
    """ Print a titled progress bar in place.
    """
    # ----------------------------------------------------------------------- #
    print('\r', title, ': |', '=' * int((a / b) * n),
          '-' if a != b else '',
          ' ' * (n - int((a / b) * n) - 1),
          '| (', a, ' / ', b, ')',
          sep = '', end = '', flush = True)


# ///////// convert_pixels_from_rgb_to_xyz / convert_pixels_from_xyz_to_rgb < #
def convert_pixels_color_space(
        rgb_pixels: List[List[ndarray]],
        pixel_conversion_function: Callable[[ndarray], ndarray]) \
        -> List[List[ndarray]]:
    """ Convert a two dimensional grid of pixels to a different color space.
    """
    # ----------------------------------------------------------------------- #
    return [[pixel_conversion_function(pixel)
             for pixel in row]
            for row in rgb_pixels]


def convert_rgb_to_xyz(
        rgb_pixel: ndarray) \
        -> ndarray:
    """ Convert color in numpy.ndarray from RGB to XYZ color space.
    """
    # ----------------------------------------------------------------------- #
    return array(convert_color(sRGBColor(*rgb_pixel),
                               XYZColor).get_value_tuple())


def convert_xyz_to_rgb(
        xyz_pixel: ndarray) \
        -> ndarray:
    """ Convert color in numpy.ndarray from XYZ to RGB color space.
    """
    # ----------------------------------------------------------------------- #
    return array(convert_color(XYZColor(*xyz_pixel,
                                        illuminant = 'd65',
                                        observer = '2'),
                               sRGBColor).get_value_tuple())


# ///////////////////////////////////////////////////////////////// k_means < #
def get_random_shuffled_range(
        n: int) \
        -> List[int]:
    """ Return randomly shuffled range of integers [0, n).
    """
    # ----------------------------------------------------------------------- #
    return random.sample(range(n), k = n)

# /////////////////////////////////////////////////////////////////////////// #
