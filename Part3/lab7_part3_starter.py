
import matplotlib.pyplot as plt

def calculate_distances(data, centroids):
    """
    Step 1: Calculate the distance between each sample and each centroid
    """
    
    
def make_clusters(distances):
    """
    Step 2: Assign each data point to it's nearest centroid
    """
    
def update_clusters(clusters, data, k, iterations):
    """
    Step 3: Average the data points in each cluster to update
    the centroids' locations and repeat for set number of iterations
    """
    

def mykmeans(data, k, centroids, iterations):
    """
    pull everything together
    
    You may randomly initialze your center
    """
    
    
# sample code for plotting the results:

def plot_clusterint(data, centroids, k):
    """
    data: 2d data points
    centroids: clustering centroids in 2d
    k: number of clusters
    
    e.g. k = 3
    """
    colors=['orange', 'blue', 'green']

    for i in range(k):

        datatmp = data[np.where(clusters == i)[0]]
        plt.scatter(datatmp[:,0], datatmp[:,1], color = colors[i])

    plt.scatter(centroids[:,0], centroids[:,1], marker='*', c='g', s=150)
    plt.show()