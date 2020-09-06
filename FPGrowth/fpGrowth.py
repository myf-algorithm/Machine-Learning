# -*- coding: utf-8 -*-
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None  # nodeLink变量用于链接相似的元素项
        self.parent = parentNode  # 指向当前节点的父节点
        self.children = {}  # 空字典，存放节点的子节点

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        """
        将树以文本形式显示
        :param ind: 1
        :return:
        """
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)  # self.children字典的值child是类似于treeNode，所以要用递归


def createTree(dataSet, minSup=1):
    """
    FP-tree构建函数
    :param dataSet: 数据集，dataSet经过处理的，它是一个字典，键是每个样本，值是这个样本出现的频数
    :param minSup: 最小支持度
    :return retTree: 构建好的树
    :return headerTable: 头指针表
    """
    headerTable = {}  # 初始化空字典作为一个头指针表
    for trans in dataSet:  # 第一次遍历数据集，统计每个元素项出现的频度
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable.keys()):  # 删除未达到最小频度的元素
        if headerTable[k] < minSup:  # 此处headerTable要取list，因为字典要进行删除del操作，字典在迭代过程中长度发生变化是会报错的
            del (headerTable[k])
    freqItemSet = set(headerTable.keys())  # 对达到最小频度的元素建立一个集合
    if len(freqItemSet) == 0: return None, None  # 若达到要求的数目为0
    for k in headerTable:  # 遍历头指针表
        headerTable[k] = [headerTable[k], None]  # 保存计数值及指向每种类型第一个元素项的指针
    retTree = treeNode('Null Set', 1, None)  # 创建只包含空集的根节点
    for tranSet, count in dataSet.items():  # 此处count为每个样本的频数
        localD = {}
        for item in tranSet:  # 通过for循环，把每个样本中频繁项集及其频数储存在localD字典中
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            # items()方法返回一个可迭代的dict_items类型，其元素是键值对组成的2-元组
            # sorted(排序对象，key，reverse),当待排序列表的元素由多字段构成时
            # 我们可以通过sorted(iterable，key，reverse)的参数key来制定我们根据哪个字段对列表元素进行排序
            # 这里key=lambda p: p[1]指明要根据键对应的值，即根据频繁项的频数进行从大到小排序
            updateTree(orderedItems, retTree, headerTable, count)  # 使用排序后的频繁项集对树进行填充
    return retTree, headerTable  # 返回树和头指针表


def updateTree(items, inTree, headerTable, count):  # 让FP树生长
    if items[0] in inTree.children:  # 首先检查事物项items中第一个元素是否作为子节点存在
        inTree.children[items[0]].inc(count)  # 则更新增加该元素项的计数
    else:  # 否则向树增加一个分支
        inTree.children[items[0]] = treeNode(items[0], count, inTree)  # 创建一个新的树节点，并更新了父节点inTree，父节点是一个类对象，包含很多特性
        if headerTable[items[0]][1] == None:  # 若原来指向每种类型第一个元素项的指针为 None，则需要更新头指针列表
            headerTable[items[0]][1] = inTree.children[items[0]]  # 更新头指针表，把指向每种类型第一个元素项放在头指针表里
        else:
            updateHeader(headerTable[items[0]][1],
                         inTree.children[items[0]])  # 更新生成链表，注意，链表也是每过一个样本，更一次链表，且链表更新都是从头指针表开始的
    if len(items) > 1:  # 仍有未分配完的树，迭代
        updateTree(items[1::], inTree.children[items[0]], headerTable,
                   count)  # 由items[1::]可知，每次调用updateTree时都会去掉列表中第一个元素，递归


def updateHeader(nodeToTest, targetNode):  # 它确保节点链接指向树中该元素项的每一个实例
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink  # 从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾，这就是一个链表
    nodeToTest.nodeLink = targetNode  # 链表链接的都是相似元素项，通过ondeLink 变量用来链接相似的元素项


def ascendTree(leafNode, prefixPath):  # 递归上溯整棵树，并把当前树节点的所有父节点处存在prefixPath列表中
    if leafNode.parent != None:  # 判断当前叶节点是否有父节点，
        prefixPath.append(leafNode.name)  # 如果有的话，向 prefixPath列表添加当前树节点，看清不是添加父节点
        ascendTree(leafNode.parent, prefixPath)  # 通过递归不断向 prefixPath列表添加当前树节点，直到没有父节点


def findPrefixPath(basePat, treeNode):
    """
    发现已给定元素项结尾的所有前缀路径
    :param basePat: 给定元素项
    :param treeNode: 在头指针表headerTable中储存着指向每种类型第一个元素项的指针
    :return condPats: 储存条件模式基的一个字典
    """
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)  # 寻找当前非空节点的前缀
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count  # 将条件模式基的个数赋给当前前缀路径，添加到字典中
        treeNode = treeNode.nodeLink  # 更换链表的下一个节点，继续寻找前缀路径
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    递归查找频繁项集
    从FP树中抽取频繁项集的三个基本步骤:
    (1) 从FP树中获得条件模式基;
    (2) 利用条件模式基,构建一个条件FP树;
    (3) 迭代重复步骤(1)步骤(2),直到树包含一个元素项为止。
    :param inTree: FP树
    :param headerTable: 头指针表
    :param minSup: 最小支持度
    :param preFix: 前缀路径（初始时一般为空，用来储存前缀路径）
    :param freqItemList: 频繁项列表（初始时一般为空，用来储存频繁项集）
    :return:
    """
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:  # 头指针表中的元素项按照其出现频率排序,从小到大（默认），从头指针表的底层开始
        # 频繁项可能是一个，也可能是多个，经过上面的步骤把单个的频繁项储存bigL
        # 寻找多个频繁项是从单个的频繁项开始的，因为一个项集是非频繁集，那么它的超级也是非频繁项集
        # 所以下面就从头指针表的最底层的单个频繁项开始，一直向上寻找。
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])  # 找见当前节点的前缀路径
        myCondTree, myHead = createTree(condPattBases, minSup)  # 将创建的条件基作为新的数据集输入到 FP树构建函数，来构建条件FP树
        if myHead != None:
            # 若myHead为非空，说明条件 FP树还可以向上缩减。
            # 说的详细一点，一开始我们通过当前节点basePat，找到条件模式基，进而创建条件FP树
            # 这棵条件FP树就是通过当前节点basePat找的频繁项（当前节点basePat本身也是单个的频繁项）
            # 接下来，我们要看若myHead为非空，找的频繁项的子集还是不是频繁项
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
            # 若myHead为非空，就把找的的频繁项条件FP树从下往上依次去一个点，看它还是不是频繁项，递归。
            # 把当前节点basePat的所有频繁项找到后，进行下一个节点的频繁项寻找


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


# import twitter
from time import sleep
import re


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


if __name__ == '__main__':
    minSup = 3
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFPtree.disp()
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)

    # 从新闻网站点击流中挖掘
    parsedDat = [line.split() for line in open('kosarak.dat').readlines()]  # 将数据集导入
    initSet = createInitSet(parsedDat)  # 对初始集合格式化
    myFPtree, myHeaderTab = createTree(initSet, 100000)
    # 构建FP树,并从中寻找那些至少被10万人浏览过的新闻报道
    myFreqList = []  # 需要创建一个空列表来保存这些频繁项集
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    print('len(myFreqList)=', len(myFreqList))
    # 看下有多少新闻报道或报道集合曾经被10万或者更多的人浏览过:
    print('myFreqList=', myFreqList)
