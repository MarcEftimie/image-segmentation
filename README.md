# Image Segmentation Project

Marc Eftimie, Anmol Sandhu, Jun Park, Venkadesh Eswaranandam

## Introduction

This project is a implementation and extension of [Julie Jiang's image segmentation project](https://github.com/julie-jiang/image-segmentation/), undertaken as a learning exercise for Olin's Discrete Mathematics course in Fall 2023. Our aim was to gain hands-on experience in image segmentation techniques and graph theory applications. We have adapted and expanded upon Julie's original work by incorporating `networkx` for graph operations and developing our own version of the augmenting path algorithm, aligning with the course's focus on discrete mathematical principles.

## Modules

1. **Augmenting Path Implementation**: [`nx_augmenting_path.py`](https://github.com/MarcEftimie/image-segmentation/blob/main/src/nx_augmenting_path.py) - This module contains the implementation of the augmenting path algorithm using `networkx`.
2. **User Input Handling**: [`nx_user_input.py`](https://github.com/MarcEftimie/image-segmentation/blob/main/src/nx_user_input.py) - Handles user inputs for selecting seed points on the image.
3. **Image to Graph Conversion**: [`nximage2graph.py`](https://github.com/MarcEftimie/image-segmentation/blob/main/src/nximage2graph.py) - Converts images into graph representations for processing.
4. **Custom Augmenting Path Algorithm**: [`our_nx_augmenting_path.py`](https://github.com/MarcEftimie/image-segmentation/blob/main/src/our_nx_augmenting_path.py) - Our custom implementation of the augmenting path algorithm.
5. **Algorithm Comparison and Benchmark**: [`benchmarking/`](https://github.com/MarcEftimie/image-segmentation/tree/main/benchmarking) - A comparison of multiple different image segmentation algorithms (K-means, N-cut, SLIC, and Mean-shift) and their runtime, using the `skimage` library.

## Installation and Usage

1. Install the required packages using `pip install -r requirements.txt`.
2. Run the program using `python main.py`.
