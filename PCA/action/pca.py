from numpy import *


def loadDataSet(fileName, delim='\t'):
    """
    加载数据集
    :param fileName: 文件名
    :param delim: 分隔符
    :return mat(datArr): 数据矩阵
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """
    pca降维
    :param dataMat: 数据矩阵
    :param topNfeat: 应用的N个特征
    :return lowDDataMat: 低维的数据矩阵
    :return reconMat: 重建的矩阵
    """
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 去均值
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算特征值和特征向量
    eigValInd = argsort(eigVals)  # 对特征值进行从小到大排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 得到topNfeat个最大的索引
    redEigVects = eigVects[:, eigValInd]  # 得到topNfeat个最大的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 把原始数据转换到新空间中
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 对将为后的数据进行重构用于调试
    return lowDDataMat, reconMat


def replaceNanWithMean():
    """
    将NaN替换为均值
    :return:
    """
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # 计算所有非NaN数据的均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # 把NaN数据替换为均值
    return datMat
