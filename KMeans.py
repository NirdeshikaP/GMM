from Kmeans_Helper import *
import matplotlib.pyplot as mpl

# Hyper parameters
k = 3
r = 5


# Groups the data into k-clusters such that each point belongs to the cluster with closest centroid.
def assign_clusters(data, k, centroids):
    clustered_data = {k:[] for k in centroids}
    for d in data:
        distance = {key: find_distance(d, v) for key, v in centroids.items()}
        assigned_cluster = min(distance, key=distance.get)
        clustered_data[assigned_cluster].append(d)

    return clustered_data


# Update the centroids such that they are equal to the mean of the points that belong to the given cluster.
def update_centroids(clustered_data):
    updated_centroids = {}
    for key in clustered_data:
        list_of_points_in_the_cluster = clustered_data[key]
        if not list_of_points_in_the_cluster:
            updated_centroids[key] = (0, 0)
        else:
            updated_centroids[key] = tuple([np.mean(x) for x in zip(*list_of_points_in_the_cluster)])

    return updated_centroids


def k_means(k):
    data = read_data()
    initial_centroids = choose_random_centroids(data=data, req_no_of_points=k)
    clustered_data = assign_clusters(data=data, k=k, centroids=initial_centroids)

    is_assignment_changed = True
    number_of_runs = 0

    while is_assignment_changed:
        is_assignment_changed = False
        updated_centroids = update_centroids(clustered_data=clustered_data)
        updated_clustered_data = assign_clusters(data=data, k=k, centroids=updated_centroids)

        for key in clustered_data:
            if clustered_data[key] != updated_clustered_data[key]:
                is_assignment_changed = True

        clustered_data = updated_clustered_data
        number_of_runs += 1

    return clustered_data, updated_centroids


def plot_graph(clustered_data, centroids, text=""):

    for key in clustered_data:
        mpl.scatter(*zip(*clustered_data[key]), s=10)

    centroid = [centroids[key] for key in centroids]
    mpl.scatter(*zip(*centroid), marker='+')
    mpl.text(0, 4, text,  fontsize=10, verticalalignment='top', horizontalalignment='center')
    mpl.show()


def main():
    for i in range(0, r):
        k_means(k)
        print('######-----######-----######-----######-----######-----######-----######')


if __name__ == "__main__":
    main()



