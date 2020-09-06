from numpy import *
import csv
import random

random.seed(21860251)


def loadDataSet():
    """
    :return postingList: 实验样本切分的词条
    :return classVec: 类别标签向量
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec  # 返回实验样本切分的词条和类别标签向量


def createVocabList(dataSet):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet: 整理的样本数据集
    :return vocabSet: 返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):  # 对比词汇表和输入的所有单词
    """
    根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    :param vocabList: createVocabList返回的列表
    :param inputSet: 切分的词条列表
    :return returnVec: 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    :param trainCategory: 训练类别标签向量，即loadDataSet返回的classVec
    :return p0Vect: 非的条件概率数组
    :return p1Vect: 侮辱类的条件概率数组
    :return pAbusive: 文档属于侮辱类的概率
    """
    numTrainDocs = len(trainMatrix)  # 文档矩阵的长度
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率

    p0Num = ones(numWords)  # 初始化概率
    p1Num = ones(numWords)
    p0Denom = 2.0  # 分母初始化
    p1Denom = 2.0  # 分母初始化

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  # 该词条的总的词数目，这样求得每个词条出现的概率 P(w1),P(w2), P(w3)...
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)  # 相除后取对数
    p0Vect = log(p0Num / p0Denom)  # 相除后取对数
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类器分类函数
    :param vec2Classify: 待分类的词条数组
    :param p0Vec: 非侮辱类的条件概率数组
    :param p1Vec: 侮辱类的条件概率数组
    :param pClass1: 文档属于侮辱类的概率
    :return 0: 属于非侮辱类
    :return 1: 属于侮辱类
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 对应元素相乘，logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):  # 朴素贝叶斯词带模型
    """
    根据vocabList词汇表，构建词袋模型
    :param vocabList: createVocabList返回的列表
    :param inputSet: 切分的词条列表
    :return returnVec: 文档向量,词袋模型
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList: # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1  # 计数加一
    return returnVec  # 返回词袋模型


def testingNB():
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  # 训练朴素贝叶斯分类器

    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))  # 执行分类并打印分类结果

    testEntry = ['stupid', 'garbage'] # 测试样本2
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))  # 执行分类并打印分类结果


def textParse(bigString):
    """
    文本切分函数
    :param bigString: 输入长的字符串
    :return: 返回列表
    """
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):  # 遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i).read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # 标记垃圾邮件，1表示垃圾文件

        wordList = textParse(open('email/ham/%d.txt' % i).read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件

    vocabList = createVocabList(docList)    # 创建词汇表，不重复
    trainingSet = list(range(50))
    testSet = []    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  # 从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值

    trainMat = []
    trainClasses = []  # 创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))  # 训练朴素贝叶斯模型

    out = open('test_result.csv', 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(['testSet', 'predict_value', 'truth'])

    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  # 测试集的词集模型
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
            print("classification error", docList[docIndex])
        test_result_list = [docIndex, classifyNB(array(wordVector), p0V, p1V, pSpam), classList[docIndex]]
        csv_write.writerow(test_result_list)  # 以覆盖方式来写入csv文件中
    print('the error rate is: ', float(errorCount) / len(testSet))


if __name__ == '__main__':
    testingNB()
    spamTest()
