import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
arrow_args = dict(arrowstyle="<-")  # 定义箭头格式


def getNumLeafs(myTree):
    """
    获取决策树叶子结点的数目
    :param myTree: 决策树
    :return numLeafs: 决策树的叶子结点的数目
    """
    numLeafs = 0  # 初始化叶子
    # firstStr = myTree.keys()[0]  # python3中myTree.keys()返回的是dict_keys,不在是list
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]  # 使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]  # 获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    获取决策树的层数
    :param myTree: 决策树
    :return maxDepth: 决策树的层数
    """
    maxDepth = 0  # 初始化决策树深度
    # firstStr = myTree.keys()[0]
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]  # 获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth  # 更新层数
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制结点
    :param nodeTxt: 结点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 结点格式
    :return: 无
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    """
    标注有向边属性值
    :param cntrPt: 用于计算标注位置
    :param parentPt: 用于计算标注位置
    :param txtString: 标注的内容
    :return: 无
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree: 决策树(字典)
    :param parentPt: 标注的内容
    :param nodeTxt: 结点名
    :return: 无
    """
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)  # 获取决策树层数
    # firstStr = myTree.keys()[0]     #the text label for this node should be this
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]# 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr] # 下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))# 不是叶结点，递归调用继续绘制
        else: # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    """
    创建绘制面板
    :param inTree: 决策树(字典)
    :return:无
    """
    fig = plt.figure(1, facecolor='white') # 创建fig
    fig.clf() # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    # createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree)) # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW # x偏移
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show() # 显示绘制结果


# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#     plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

def retrieveTree(i):
    """
    预留的一棵树字典
    :param i: 索引
    :return:
    """
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


if __name__ == '__main__':
    import trees
    import treePlotter

    dataSet, labels = trees.createDataSet()
    myTree = trees.createTree(dataSet, labels)
    print(myTree)
    treePlotter.createPlot(myTree)

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    print(lensesTree)
    treePlotter.createPlot(lensesTree)

    trees.storeTree(lensesTree, 'test.txt')
    trees.grabTree('test.txt')
