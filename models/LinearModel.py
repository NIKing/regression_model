"""
定义一个只有两层的前馈神经网络
"""
class LinearModel():
    def __init__(self):
        self.parameter = {}
        self.feature_map = {}

    def __call__(self, features = None):
        # 在这里，我有点不太明白，如果设置特征矩阵，那么这个特征矩阵应该存放的是什么呢？全部的特征形成的矩阵还是当前特征形成的特征矩阵？
        
        for i, feature in enumerate(features):
            self.add_feature(feature)
            
            self.feature_matrix[i] = feature

        return self.forward() 

    def add_feature(self, feature):
        ids = []
        if not feature in self.feature_map:
            self.feature_map[feature] = len(self.feature_map)

        ids.append(self.feature_map[feature])

        return ids
        
    def forward(self):
        # 放射函数中(w*x) 权重值是parameter，但是特征值怎么转换呢？在这里特征值是直接给的，但是在中文分词中的特征值？
        # 在中文分词中，w * x 代表：权重向量与特征向量做点积后，得到一个标量
        # 但是，在这里貌似直接可以使用输入的特征值？
        outputs = []
        for i, ids in enumerate(self.feature_matrix):
            
            source = 0
            for id in ids:
                source += self.parameter[id]

            outpus[i].append(source)

        return
