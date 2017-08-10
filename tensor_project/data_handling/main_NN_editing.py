import numpy as np
import time
import cx_Oracle
from tensor_project.xray_image_to_list_junho_ver import OraDB

# training_epochs = 20
batch_size = 200
training_set_idx = np.random.permutation(7470)[:6000]
test_set_idx = np.random.permutation(7470)[6001:]



#todo list를 sampling 하는 법은
#todo random.sample(L,10)



def data_setting(data):
    # x : 데이터, y : 라벨
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 10])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return x, y




def clob2List(imageVal):
    return  list(map(int,imageVal.split(',')))


def classfiyDataSet():
    cur = OraDB.prepareCursor()
    xrayTabel = cur.execute('select * from xray_data')
    training_set_idx = np.random.permutation(7470)[:6000]
    test_set_idx = np.random.permutation(7470)[6001:]
    input_list = []
    for row in xrayTabel:
        _, gray, label = list(row)
        gray = cx_Oracle.LOB.read(gray)  # str형태
        gray = clob2List(gray)  # [255,322,] list형태
        input_list.append(gray+[int(label)])  # list형태로 넣는다. [int, list, int]
    trsArray = np.array(input_list)  # array( [ [],[] ] )
    training_list = trsArray[training_set_idx]
    test_list = trsArray[test_set_idx]
    print('훈련데이터 shape = ',training_list.shape)
    print('검증데이터 shape = ', test_list.shape)

    OraDB.releaseConn() #con객체를 닫아준다.

    return training_list, test_list



training_list, test_list = classfiyDataSet()
