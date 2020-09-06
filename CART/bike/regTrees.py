# -*- coding: utf-8 -*-
from numpy import *


def loadDataSet(fileName):
    """
    将每行映射成浮点数
    :param fileName: 输入文件
    :return dataMat: 数据集
    """
    dataMat = []  # 最后一列为label
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(list(fltLine))  # map函数返回的是map对象，而不是list
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    数据集进行二分
    :param dataSet: 数据集
    :param feature: 要分割的特征
    :param value: 分割的阈值
    :return mat0, mat1: 分割后的数据集
    """
    mat0 = dataSet[nonzero(dataSet[:, feature] > value), :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1


def regLeaf(dataSet):
    """
    CART回归树的生成叶子节点函数
    :param dataSet: 输入数据集
    :return mean: 目标变量的均值
    """
    return mean(dataSet[:, -1])


def regErr(dataSet):
    """
    最好特征选择依据：
    误差-样本y的均方差乘上样本总数=总方差，作为当前样本情况下的混乱程度
    :param dataSet: 数据集
    :return: 误差
    """
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def linearSolve(dataSet):
    """
    模型树的叶子节点生成函数
    :param dataSet: 数据集
    :return ws, X, Y: 系数
    """
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)  # 线性回归方程
    return ws, X, Y


def modelLeaf(dataSet):
    """
    创建线性模型
    :param dataSet: 数据集
    :return: 系数
    """
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    模型树的误差计算
    :param dataSet: 数据集
    :return: 误差
    """
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    选择最佳特征和阈值
    :param dataSet: 数据集
    :param leafType: 叶子类型，函数
    :param errType: 误差类型，函数
    :param ops: (容许的误差下降值，切分的最小样本数)
    :return bestIndex, bestValue: 最优的特征，最优的值
    """
    tolS = ops[0]  # 容许的误差下降值-防止过拟合-预剪枝
    tolN = ops[1]  # 切分的最小样本数-防止过拟合-预剪枝
    # 如果y取值全部相等，则不需要在划分了，直接返回
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 退出的条件1
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)  # 计算当前划分下的混乱程度，随着不断的划分，混乱程度应该是下降
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):  # 对每一个特征，n-1是特性个数
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):  # 对每个特征所对应的特征值，也就是划分阈值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # 基于该阈值进行样本二分
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 不合适的划分方法
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 划分后，混乱程度几乎没有下降，则不划分了
        return None, leafType(dataSet)  # 退出的条件2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 正式开始划分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 退出的条件3
        return None, leafType(dataSet)
    return bestIndex, bestValue  # 返回特征索引和对应的划分阈值


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    递归创建树
    :param dataSet: 数据集
    :param leafType: 叶子类型，函数
    :param errType: 误差类型，函数
    :param ops: (容许的误差下降值，切分的最小样本数)
    :return retTree: 创建好的树
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:  # 停止条件
        return val
    retTree = {}
    retTree['spInd'] = feat  # 特征索引
    retTree['spVal'] = val  # 特征索引列对应的特征划分阈值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 开始递归
    retTree['right'] = createTree(rSet, leafType, errType, ops)  # 开始递归
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    """
    从上往下，递归查找两个叶子节点
    :param tree: 输入的树
    :return: 均值
    """
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    后剪枝操作
    :param tree: 待剪枝树
    :param testData: 剪枝所需测试数据(一般是验证集)
    :return: 剪枝后的树
    """
    if shape(testData)[0] == 0: return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # 经过N次迭代，找到了叶子节点，开始合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 没有合并前的误差和
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        # 合并后的误差和
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


# 回归树评估函数
def regTreeEval(model, inDat):
    return float(model)


# 模型树评估函数
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


# 测试数据输入的入口函数
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    # test1
    # myDat=loadDataSet("ex00.txt")
    # myMat=mat(myDat)
    # print(createTree(myMat))

    # tes2t
    # myDat1=loadDataSet("ex0.txt")
    # myMat1=mat(myDat1)
    # print(createTree(myMat1))

    # test3-后剪枝
    # myDat2=loadDataSet("ex2.txt")
    # myMat2=mat(myDat2)
    # tree=createTree(myMat2, ops=(0, 1))
    # print(tree)
    # myDatTest = loadDataSet("ex2test.txt")
    # myMatTest = mat(myDatTest)
    # print(prune(tree,myMatTest))

    # test4-模型树
    # myDat3 = loadDataSet("exp2.txt")
    # myMat3 = mat(myDat3)
    # print(createTree(myMat3, modelLeaf, modelErr, ops=(1, 10)))

    # test5-模型比较
    # 创建回归树
    trainMat = mat(loadDataSet("bikeSpeedVsIq_train.txt"))
    testMat = mat(loadDataSet("bikeSpeedVsIq_test.txt"))
    myTree = createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])  # 预测
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    # 创建模型树
    myTree = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])

    # 创建线性回归模型
    ws, X, Y = linearSolve(trainMat)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
