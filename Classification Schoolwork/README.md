
### 一、数据分析

1.对于给定的200维数据，先粗略地查看其数据分布
常用函数：data.describe()、data.info()、

2.因维数过高，所以我们考虑对数据做相关性分析，以便去掉不相关或对结果影响不大的列，

这里我们常用函数库seaborn、matplotlib来绘制图像以便于可视化

具体代码：dataicorr= datai.corr()；
               sns.heatmap(dataicorr)

从可视化后的图片可以看出，几乎所有的特征都与结果有明显的相关性

### 二、数据降维

降维的方法有mds，svd，pca等，这里我们选用最常见的pca，
pca参数的确定：
                pca调参的过程我选的是暴力挑参法，经试验结果表明，pca默认的参数效果最好

### 三、数据预处理

读入数据后，分离数据属性集和标签，以便于训练和数据预处理，标签要进行dataframe编码

### 四、模型的选择

集成学习的结果要优于其他学习模型，调参后gbdt在此次问题中的表现优于其他模型，所以在此次实际问题中选取的模型为调参后的gbdt


```python
参数选择的结果为：learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None, 
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None, 
    warm_start=False
```
