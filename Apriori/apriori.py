from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    """
    创建第一个候选项集的列表C1
    :param dataSet: 数据集
    :return: 对C1中的每个项构建一个不变集合
    """
    C1 = []  # 创建一个空列表C1，用于存储所有不重复的项值
    for transaction in dataSet:  # 遍历数据集的每一条交易记录
        for item in transaction:  # 遍历记录中的每一个项
            if not [item] in C1:  # 如果某个物品项没有出现在C1中
                C1.append([item])  # 添加到C1中
    C1.sort()  # 对大列表进行排序
    return list(map(frozenset, C1))  # 对C1中的每个项构建一个不变集合


def scanD(D, Ck, minSupport):
    """
    从C1生成L1
    :param D: 数据集
    :param Ck: 候选项集合列表Ck
    :param minSupport: 感兴趣项集的最小支持度
    :return supportData: 最频繁项集的支持度
    """
    ssCnt = {}  # 创建空字典
    for tid in D:  # 遍历每个样本
        for can in Ck:  # 遍历每个项集
            if can.issubset(tid):  # 统计频数
                if not can in ssCnt:
                    ssCnt[can] = 1  # 不含设为1
                else:
                    ssCnt[can] += 1  # 有则计数加1
    numItems = float(len(D))  # 数据集大小
    retList = []  # L1初始化
    supportData = {}  # 记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems  # 转换为频率
        if support >= minSupport:  # 如果大于最小支持度，则保持
            retList.insert(0, key)  # 将字典元素添加到retList中L1中
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    创建Ck
    :param Lk: 频繁项集列表LK
    :param k: 项集元素个数K
    :return retList: Ck
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):  # 两两组合遍历
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # #若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(Lk[i] | Lk[j])  # 将两个集合进行合并
    return retList


def apriori(dataSet, minSupport=0.5):
    """
    apriori算法的核心程序
    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return L: L1,L2,...
    """
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):  # 创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck = aprioriGen(L[k - 2], k)  # Ck
        Lk, supK = scanD(D, Ck, minSupport)  # 扫描数据集，从Ck得到Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):  # supportData is a dict coming from scanD
    """
    关联规则生成
    :param L: 频繁项集列表
    :param supportData: 包含那些频繁项集支持数据的字典
    :param minConf: 最小可信度阈值
    :return bigRuleList: 包含可信度的规则列表
    """
    bigRuleList = []  # 存储所有的关联规则
    for i in range(1, len(L)):  # #只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        for freqSet in L[i]:  # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
            H1 = [frozenset([item]) for item in freqSet]  # 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):  # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    """
    针对项集中只有两个元素时，计算可信度
    :param freqSet: 频繁项集
    :param H: 可以出现在规则右部的元素列表H
    :param supportData: 包含那些频繁项集支持数据的字典
    :param brl: 包含可信度的规则列表
    :param minConf: 最小可信值
    :return:
    """
    prunedH = []  # 返回一个满足最小可信度要求的规则列表
    for conseq in H:  # 后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 可信度计算，结合支持度数据
        if conf >= minConf:
            # 如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))  # 添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)  # 同样需要放入列表到后面检查
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    """
    从最初的项集中生成更多的关联规则
    :param freqSet: 频繁项集
    :param H: 可以出现在规则右部的元素列表H
    :param supportData: 包含那些频繁项集支持数据的字典
    :param brl: 包含可信度的规则列表
    :param minConf: 最小可信值
    :return:
    """
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m + 1)  # 存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 计算可信度
        if (len(Hmp1) > 1):  # 满足最小可信度要求的规则列表多于1，则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print(itemMeaning[item])
        print("           -------->")
        for item in ruleTup[1]:
            print(itemMeaning[item])
        print("confidence: %f" % ruleTup[2])


if __name__ == '__main__':
    # test1-L1
    dataSet = loadDataSet()
    print(dataSet)
    C1 = createC1(dataSet)
    print(C1)
    D = list(map(set, dataSet))
    print(D)
    L1, support = scanD(D, C1, 0.5)
    print("频繁项集：", L1, "\n支持度", support)

    # test2-完整程度
    dataSet = loadDataSet()
    L, supportData = apriori(dataSet)
    print("频繁项集：", L, "\n支持度", supportData)

    # test3-关联规则生成
    rules = generateRules(L, supportData, minConf=0.5)
    print("关联规则为：", rules)


