
### 数据集iris的导入及处理


```python
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

iris = datasets.load_iris()
labels = iris.target
features = iris.data
#pca降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(features)
irisdata= pca.transform(features)
```

### 原始数据可视化


```python
features_1 = plt.scatter(irisdata[0:50, 0], irisdata[0:50, 1],marker='o')
features_2 = plt.scatter(irisdata[51:100, 0], irisdata[51:100, 1],marker='ve')
features_3 = plt.scatter(irisdata[101:149, 0], irisdata[101:149, 1],marker='vi')
plt.legend(handles=[features_1, features_2,features_3], labels=['setosa', 'versicolor', 'virginica'],  loc='best')
plt.plot()
```

### 一、markov聚类的过程

算法实现步骤：


1.输入一个无向图，Expansion的幂e（此项目用expand_factor）和Inflation的参数r（inflate_factor）


2.创建邻接矩阵

3.对每个结点添加自循环

4.标准化矩阵（每个元素除以所在列的所有元素之和）

5.计算矩阵的第e次幂

6.用参数r(此项目中用inflate_factor替代r)对求得的矩阵进行Inflation处理，

7.重复第5步和第6步，直到状态稳定不变（收敛）

8.把最终结果矩阵转换成聚簇。

代码实现：


```python
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
#inflation矩阵
def inflate(A, inflate_factor):
    return normalize(np.power(A, inflate_factor))
#扩展矩阵
def expand(A, expand_factor):
    return np.linalg.matrix_power(A, expand_factor)

def add_diag(A, mult_factor):
    return A + mult_factor * np.identity(A.shape[0])
#markov聚类
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


```

### 参数（expand_factor和inflate_factor）的选择：
            使用格点搜索法，试验结果表明e=2,r=2时效果较好

### 聚类的评估
def acc():
    return accuracy_score(labels,lable) 


```python

```
