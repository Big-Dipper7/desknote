# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:59:39 2019

@author: Administrator
"""

import numpy as np


def Generate_Systemdata(k, dim, dim_A, num):

    '''
    k: 系统包的个数 k = 4,9,16
    dim:数据矩阵的维度
    return: 两个已分块的矩阵
    '''
    list_A = list()
    list_x = list()
    list_choose = list()

#    list_B.append(np.random.rand(dim, 1)*100)

    list_x.append(np.ones((dim, 1)))


    for i in range(k):
        list_A.append((i + 1)*np.ones((dim, dim)))

    for i in range(num):
        list_choose.append(i)


    '''
    for i in range(k):
        list_A.append(100*np.random.rand((dim, dim_A)))
    '''


    return list_A, list_x, list_choose


#循环移位函数
def shift_matrix(lst, a):
    '''
    A:待移位矩阵
    a：移位的位数
    return:已移位的矩阵
    '''

    return lst[-a:] + lst[ :-a]


#系统包
def mul(A, B):
    return np.dot(A, B)


def Generate_Systemmatrix(list_A, list_B):

    #计算系统包
    matrix_data = list()
    for i in range(len(list_A)):
        for j in range(len(list_B)):

            result_Ax = mul(list_A[i], list_B[j])

            matrix_data.append(result_Ax)

    return matrix_data



#两个不同维度的矩阵相加或相减
def matrix_mat(matrix1):
    #先求出两个矩阵的维度
    xshape = list()
    yshape = list()

    for i in range(len(matrix1)):

        xshape.append(matrix1[i].shape[0])
        yshape.append(matrix1[i].shape[1])
    #求出最大维度
    max_x= max(xshape)
    max_y= max(yshape)

    #按最大维度对小的矩阵进行补零
    for i in range(len(matrix1)):

        zero_x = np.zeros((xshape[i], max_y - yshape[i]))
        zero_y = np.zeros((max_x - xshape[i], max_y))

        matrix1[i] = np.concatenate((matrix1[i], zero_x), axis = 1)
        matrix1[i] = np.concatenate((matrix1[i], zero_y), axis = 0)
    #判断相加或相减

    result = np.zeros((max_x, max_y))
    for i in range(len(matrix1)):
        result = result + matrix1[i]

    return result


#编码函数   仅限于A = [A1; A2]情况
def Code(shift, A, list_x):
    A_a = list()
    for i in range(len(shift)):

        zero = np.zeros([shift[i], A[i].shape[1]])
        A_a.append(np.concatenate((zero, A[i]), axis = 0))

    #print(A_a)
    #print(B_b)

    AddResult_A = matrix_mat(A_a)

    result = mul(AddResult_A, list_x)
    result = result.reshape(AddResult_A.shape[0], list_x[0].shape[1])

    return result


def Generate_Codematrix(list_A, list_x, Shift):

    matrix_data = list()
    for i in range(len(Shift)):
        matrix_data.append(Code(Shift[i], list_A, list_x))

    return matrix_data



def decodeSystemPackage(receive_data,
                        CodePackageShift,
                        DecodedResults,
                        ValidationPackage,
                        DecodedResults_shape,
                        CountSystemPackage,
                        receive_data_index):
    '''
    #receive_data：接收到的数据包，包括系统包和编码包
    #receive_data_index：接收到的数据包的索引
    #CodePackageShift：编码包的移位矩阵
    #dim：数据维度
    #CountSystemPackage：系统包的个数
    #DecodedResults：待恢复的数据
    #ValidationPackage：验证作用的系统包
    '''
    SystemData = receive_data[: CountSystemPackage]#取出系统包
    CodeData = receive_data[CountSystemPackage: ]#取出编码包

    #把系统包先放入其中
    for i in range(len(SystemData)):
        DecodedResults[receive_data_index[i]] = SystemData[i]


    #此函数实现对None地地方填0操作
    DecodedResults, Is_None = Fill_None(DecodedResults)

    for i in range(10000):
        #用编码包减去移位后的系统包
        IterData = IterDifferAll(CodeData,              #全部编码数据包
                                 DecodedResults,        #初步解码后的结果
                                 CodePackageShift)      #编码包的移位矩阵

        #F函数，功能为寻找暴露位
        DecodedResults = F_fuction(Is_None,             #True 和 false矩阵，已恢复的数据为false，未恢复的数据为True
                                   IterData,            #编码包减去系统包和一维解码的结果
                                   CodePackageShift,    #
                                   DecodedResults,      #
                                   DecodedResults_shape)

        #F函数的前提，重新填写True和False
        Is_None, CountTrue = Refill_One(DecodedResults,
                                        Is_None)

        if CountTrue == 0:
            #print("迭代", i+1, "次，剩余", CountTrue, "个数据。")
            #TotalResult = 0
            break

        #print(DecodedResults)
        #print("迭代", i+1, "次，剩余", CountTrue, "个数据。")
    '''
    #print(DecodedResults)
    TotalResult = Validation(ValidationPackage,     #验证数据包
                             DecodedResults)        #恢复数据的结果

    if(TotalResult == 0):
        #print("验证成功，该数据包可解码")
        return TotalResult
    '''

#判断验证数据包和解码的数据包是否相等
def Validation(ValidationPackage, DecodedResults):
    TotalResult = 0
    for i in range(len(ValidationPackage)):
        for j in range(len(ValidationPackage[i])):
            for k in range(len(ValidationPackage[i][j])):
                    result  = round(ValidationPackage[i][j][k] - DecodedResults[i][j][k], 5)
                    if result != 0:
                        #求出不相等的个数
                        TotalResult = TotalResult + 1
    return TotalResult


#对初步恢复的数据包填True和False，已恢复填False，未恢复填True
def Refill_One(DecodedResults, Is_None):
    CountTrue = 0
    for i in range(len(DecodedResults)):
        for j in range((len(DecodedResults[i]))):
            for k in range((len(DecodedResults[i][j]))):
                if round(DecodedResults[i][j][k], 5) == 0:
                    Is_None[i][j][k] = True
                    CountTrue = CountTrue +1
                else:
                    Is_None[i][j][k] = False
    return Is_None, CountTrue


#此函数实现对None地地方填0操作
def Fill_None(DecodedResults):
    #找出空值的地方，返回bool类型

    Is_None = list()

    for i in DecodedResults:
        Is_None.append(np.isnan(i))

    for i in range(len(DecodedResults)):
        for j in range(len(DecodedResults[i])):
            for k in range(len(DecodedResults[i][j])):
                if (Is_None[i][j][k] == True):
                    DecodedResults[i][j][k] = 0
    return DecodedResults, Is_None

#迭代，多个编码包
def IterDifferAll(CodeData,
                  SystemData,
                  CodePackageShift):

    mid_result = list()

    for i in range(len(CodeData)):

        mid_result.append(differ(CodeData[i],
                                 SystemData,
                                 CodePackageShift[i]))

    return mid_result


#一个编码包减去多个系统包后的结果
def differ(CodeData,
           SystemData,
           CodePackageShift):

    result = list()

    for i in range(len(SystemData)):

        result = ZeroShift(SystemData[i], CodePackageShift[i])

        CodeData = matrix_dif(CodeData,
                              result,
                              add = 0)

    return CodeData

#两个不同维度的矩阵相加或相减
def matrix_dif(matrix1, matrix2, add):
    #先求出两个矩阵的维度
    matrix = [matrix1, matrix2]

    x1, y1 = matrix1.shape
    x2, y2 = matrix2.shape

    x = [x1, x2]
    y = [y1, y2]
    #求出最大维度
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    #按最大维度对小的矩阵进行补零
    for i in range(2):

        zero_x = np.zeros((x[i], max_y - y[i]))
        zero_y = np.zeros((max_x - x[i], max_y))

        matrix[i] = np.concatenate((matrix[i], zero_x), axis = 1)
        matrix[i] = np.concatenate((matrix[i], zero_y), axis = 0)
    #判断相加或相减
    if add == 1:
        return matrix[0] + matrix[1]
    else:
        return matrix[0] - matrix[1]



#根据移位矩阵填充系统包
def ZeroShift(data, shift):

    x, y = data.shape

    zero_D = np.zeros((shift, 1))

    data = np.concatenate((zero_D, data), axis = 0)

    return data

#对D, R移位进行选择
def Shift_Choose(CodePackageShift, index):

    #shift = CodePackageShift

    shift_len = len(CodePackageShift)

    shift_index = list()

    for i in range(shift_len):
        for j in range(shift_len):

            shift_index.append([i,j])

    D = CodePackageShift[shift_index[index][0]]
    R = CodePackageShift[shift_index[index][1]]

    return [D, R]


#F函数，功能为寻找暴露位
def F_fuction(Is_None, CodeData, CodePackageShift, DecodedResults, DecodedResults_shape):

    #Is_None = Is_None.astype(int)

    for i in range(len(Is_None)):

        Is_None[i] = Is_None[i].astype(int)


    #Mid_DecodedResults = list()
    for i in range(len(CodeData)):
        #比较哪个地方有相同的“1”
        Mid_DecodedResults = CompareValue(CodeData[-1 -i], #编码包的位置从最大到最小
                                          Is_None,
                                          CodePackageShift[-1 - i], #取出编码矩阵
                                          DecodedResults,
                                          DecodedResults_shape)

        for i in range(len(DecodedResults)):
            #判断是否应该相加，因为某个位置重复为暴露位
            DecodedResults[i] = Judge_add(DecodedResults[i],
                                          Mid_DecodedResults[i])
        Mid_DecodedResults = list()

    return DecodedResults

#DecodedResults位置为0的地方可以相加
def Judge_add(DecodedResults, Mid_DecodedResults):

    for i in range(len(DecodedResults)):
        for j in range(len(DecodedResults[i])):
            if round(DecodedResults[i][j], 5) == 0:
                DecodedResults[i][j] = DecodedResults[i][j] + Mid_DecodedResults[i][j]

    return DecodedResults



def CompareValue(CodeData, Is_None, CodePackageShift, DecodedResults, DecodedResults_shape):

    result = list()
    #对Is_None填充1和0
    for i in range(len(Is_None)):
        result.append(ZeroFill(ZeroShift(Is_None[i],
                                         CodePackageShift[i]),    #选择编码矩阵
                                         CodeData))
    #计算所有矩阵的总和
    Sum_result = np.sum(result, axis=0)
    #把总和矩阵中的1找出来，只有1是需要的，2,3,4不需要
    Store_F_Value = list()
    Store_F_Value.append((Sum_result == 1).astype(int))

    #存储经过F函数后的结果，该结果为1的地方为对应系统包的暴露位
    Store_AfterF_Value = list()
    for i in range(len(result)):
        #恢复正确的移位
        Mid_Result = Recover_Shift(JudgeToOne(Store_F_Value[0], result[i]) * CodeData,
                                   CodePackageShift,
                                   i,
                                   DecodedResults_shape[i])
        #收集数据
        Store_AfterF_Value.append(Mid_Result)

    return Store_AfterF_Value



#填充矩阵，使两个维度一样
def ZeroFill(data, CodePackage):

    matrix = [data, CodePackage]

    x1, y1 = data.shape
    x2, y2 = CodePackage.shape

    x = [x1, x2]
    y = [y1, y2]

    max_x = max(x1, x2)
    max_y = max(y1, y2)

    for i in range(2):

        zero_x = np.zeros((x[i], max_y - y[i]))
        zero_y = np.zeros((max_x - x[i], max_y))

        matrix[i] = np.concatenate((matrix[i], zero_x), axis = 1)
        matrix[i] = np.concatenate((matrix[i], zero_y), axis = 0)
    return matrix[0]

#判断矩阵是否为1，此函数可以找出和编码包相同的“1”,这个”1“可以为对应系统包的暴露位
def JudgeToOne(Store_F_Value, result):

    for i in range(len(Store_F_Value)):
        for j in range(len(Store_F_Value[i])):
                if Store_F_Value[i][j] == result[i][j] and result[i][j] == 1:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
    return result


#恢复正确的移位
def Recover_Shift(Mid_Result, CodePackageShift, i, DecodedResults_shape):

    ShiftIndex = CodePackageShift[i]

    return recover(Mid_Result, ShiftIndex, DecodedResults_shape)

#从已经解码的含移位的数据包中恢复系统包
def recover(result, shift_index, shape):

    Row = shape[0]

    result = result[shift_index: shift_index+Row]

    return result



