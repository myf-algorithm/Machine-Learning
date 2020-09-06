from numpy import *
from numpy import linalg as la


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def ecludSim(inA, inB):
    """
    计算欧式距离
    :param inA: 输入向量A
    :param inB: 输入向量B
    :return: 相似度
    """
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    """
    计算皮尔逊相关系数
    :param inA: 输入向量A
    :param inB: 输入向量B
    :return: 相似度
    """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """
    计算余弦相似度
    :param inA: 输入向量A
    :param inB: 输入向量B
    :return: 相似度
    """
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    """
    计算用户对物品的评分值
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号
    :return: 相似度归一化
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]  # 寻找两个用户都评级的物品
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def svdEst(dataMat, user, simMeas, item):
    """
    基于SVD计算用户对物品的评分值
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号
    :return: 相似度归一化
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])  # 建立对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # 构建转换后的物品
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    推荐函数
    :param dataMat: 数据矩阵
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :return: 寻找前N个未评级物品
    """
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 寻找未评级的用户
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]  # 寻找前N个未评级物品


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end=' ')
            else:
                print(0, end=' ')
        print('')


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)

    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    print(SigRecon)
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    # 图像压缩
    imgCompress()

    # 测试相似度计算函数
    myMat = mat(loadExData())
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))
    print(pearsSim(myMat[:, 0], myMat[:, 4]))

    # 使用standEst进行推荐
    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print(recommend(myMat, 2))
    print(recommend(myMat, 2, simMeas=ecludSim))
    print(recommend(myMat, 2, simMeas=pearsSim))

    # 使用svdEst进行推荐
    myMat = mat(loadExData2())
    print(recommend(myMat, 1, estMethod=svdEst))
    print(recommend(myMat, 1, simMeas=ecludSim, estMethod=svdEst))
    print(recommend(myMat, 1, simMeas=pearsSim, estMethod=svdEst))

