
import sys
import numpy as np
import time
from optparse import OptionParser
import logging
#构建G矩阵
def get_graph(csv_filename):
    import networkx as nx

    M = []
    for r in open(csv_filename):
        r = r.strip().split(",")
        M.append(list(map(lambda x: float(x.strip()), r)))

    G = nx.from_numpy_matrix(np.matrix(M))
    return np.array(M), G
#构建相似度矩阵
from sklearn import datasets
iris = datasets.load_iris()
features = iris.data
A =[]
n=150

def grapha(features):
    n=150
    A =[]
    for i in features:
        for j in features:
            distance = np.abs(i[0]-j[0])+np.abs(i[1]-j[1])+np.abs(i[2]-j[2])+np.abs(i[3]-j[3])  
            A.append(float(distance))  
    A = np.array(A)
    A = s_array.reshape(n,n) 
    return A


#标准化矩阵
def normalize(A):
    column_sums = A.sum(axis=0)
    new_matrix = A / column_sums[np.newaxis, :]
    return new_matrix
#inflation矩阵
def inflate(A, inflate_factor):
    return normalize(np.power(A, inflate_factor))
#扩展矩阵
def expand(A, expand_factor):
    return np.linalg.matrix_power(A, expand_factor)
#自环
def add_diag(A, mult_factor):
    return A + mult_factor * np.identity(A.shape[0])

def get_clusters(A):
    clusters = []
    for i, r in enumerate((A>0).tolist()):
        if r[i]:
            clusters.append(A[i,:]>0)

    clust_map  ={}
    for cn , c in enumerate(clusters):
        for x in  [ i for i, x in enumerate(c) if x ]:
            clust_map[cn] = clust_map.get(cn, [])  + [x]
    return clust_map
#可视化
def draw(G, A, cluster_map):
    import networkx as nx
    import matplotlib.pyplot as plt

    clust_map = {}
    for k, vals in cluster_map.items():
        for v in vals:
            clust_map[v] = k

    colors = []
    for i in range(len(G.nodes())):
        colors.append(clust_map.get(i, 100))

    pos = nx.spring_layout(G)

    from matplotlib.pylab import matshow, show, cm
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos,node_size = 200, node_color =colors , cmap=plt.cm.Blues )
    nx.draw_networkx_edges(G,pos, alpha=0.5)
    matshow(A, fignum=1, cmap=cm.gray)
    plt.show()
    show()

#停止条件
def stop(M, i):

    if i%5==4:
        m = np.max( M**2 - M) - np.min( M**2 - M)
        if m==0:
            logging.info("Stop at iteration %s" % i)
            return True

    return False

#马尔可夫聚类
def mcl(M, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    M = add_diag(M, mult_factor)
    M = normalize(M)

    for i in range(max_loop):
        logging.info("loop %s" % i)
        M = inflate(M, inflate_factor)
        M = expand(M, expand_factor)
        if stop(M, i): break

    clusters = get_clusters(M)
    return M, clusters

def networkx_mcl(G, expand_factor = 2, inflate_factor = 2, max_loop = 10 , mult_factor = 1):
    import networkx as nx
    A = nx.adjacency_matrix(G)
    return mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)

def print_info(options):
    print("-" * 60)
    print("MARKOV CLUSTERING:")
    print("-" * 60)
    print("  expand_factor: %s" % options.expand_factor)
    print("  inflate_factor: %s" % options.inflate_factor)
    print("  mult factor: %s" % options.mult_factor)
    print("  max loops: %s\n" % options.max_loop)

def get_options():
    usage = "usage: %prog [options] <input_matrix>"
    parser = OptionParser(usage)
    parser.add_option("-e", "--expand_factor",
                      dest="expand_factor",
                      default=2,
                      type=int,
                      help="expand factor (default: %default)")
    parser.add_option("-i", "--inflate_factor",
                      dest="inflate_factor",
                      default=2,
                      type=float,
                      help="inflate factor (default: %default)")
    parser.add_option("-m", "--mult_factor",
                      dest="mult_factor",
                      default=2,
                      type=float,
                      help="multiply factor (default: %default)")
    parser.add_option("-l", "--max_loops",
                      dest="max_loop",
                      default=60,
                      type=int,
                      help="max loops (default: %default)")
    parser.add_option("-o", "--output", metavar="FILE", 
                      help="output (default: stdout)")

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=True,
                      help="verbose (default: %default)")
    parser.add_option("-d", "--draw-graph",
                      action="store_true", dest="draw", default=False,
                      help="show graph with networkx (default: %default)")
    

    (options, args) = parser.parse_args()

    try:
        filename = args[0]
    except:
        raise Exception('input', 'missing input filename')


    return options, filename
#聚类的输出
def clusters_to_output(clusters, options):
    if options.output and len(options.output)>0:
        f = open(options.output, 'w')
        for k, v in clusters.items():
            f.write("%s|%s\n" % (k, ", ".join(map(str, v)) ))
        f.close()
    else:
        print("Clusters:")
        for k, v in clusters.items():
            print('{}, {}'.format(k, v))


if __name__ == '__main__':
    M, G = get_graph(filename)

    print(" number of nodes: %s\n" % M.shape[0])

    print("{}: {}".format(time.time(), "evaluating clusters..."))
    M, clusters = networkx_mcl(G, expand_factor = options.expand_factor,
                               inflate_factor = options.inflate_factor,
                               max_loop = options.max_loop,
                               mult_factor = options.mult_factor)
    print("{}: {}".format(time.time(), "done\n"))

    clusters_to_output(clusters, options)

    if options.draw:
        print("{}: {}".format(time.time(), "drawing..."))
        draw(G, M, clusters)
        print("{}: {}".format(time.time(), "done"))
        
#聚类的评估
def acc():
    return accuracy_score(labels,lable) 
        
#引入iris'数据集进行聚类        
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

iris = datasets.load_iris()
labels = iris.target
features = iris.data
#pca降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(features)
pca_features= pca.transform(features)
#可视化原始iris数据集
features_1 = plt.scatter(pca_features[0:50, 0], pca_features[0:50, 1],marker='o')
features_2 = plt.scatter(pca_features[51:100, 0], pca_features[51:100, 1],marker='ve')
features_3 = plt.scatter(pca_features[101:149, 0], pca_features[101:149, 1],marker='vi')
plt.legend(handles=[features_1, features_2,features_3], labels=['setosa', 'versicolor', 'virginica'],  loc='best')
plt.plot()
#对原始数据集进行mcl聚类
M, clusters = networkx_mcl(features)

