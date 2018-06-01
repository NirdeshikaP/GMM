import numpy as np
import math


# Read data from the txt file and split them to list of co-ordinate points [(x1,y1),(x2,y2)..]
def read_data():
    lines = open('GMM_dataset.txt', 'r').read().split('\n')
    data = [tuple(map(float,line.strip().split("  "))) for line in lines]
    return data


# Randomly choose req_no_of_points number of initial centroids/means
def choose_random_centroids(data, req_no_of_points):
    total_no_of_points = data.__len__()

    random_centroids_index = np.random.randint(low=0, high=total_no_of_points-1, size=req_no_of_points)

    random_centroids = {}
    for i, c in enumerate(random_centroids_index):
        random_centroids[i] = data[c]

    return random_centroids


def find_distance(point1, point2):
    distance = math.hypot(point1[0]-point2[0], point1[1]-point2[1])
    return distance


def find_wcss(clustered_data, centroid):
    total_wcss = 0
    for key in centroid:
        distances = map(lambda x: find_distance(x, centroid[key]), clustered_data[key])
        total_wcss += sum(distances)
    return total_wcss