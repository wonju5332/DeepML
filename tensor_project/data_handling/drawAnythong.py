import numpy as np
import cx_Oracle

class OraDB:
    INFO = 'scott/tiger@localhost:1522/orcl3'
    ADMIN = None

    @classmethod
    def createConn(cls, info):
        OraDB.ADMIN = cx_Oracle.connect(info)
        print('DB 커넥션 객체가 생성되었습니다.')

    @classmethod
    def prepareCursor(cls):
        return OraDB.ADMIN.cursor()

    @classmethod
    def dbCommit(cls):
        return OraDB.ADMIN.commit()

    @classmethod
    def releaseConn(cls):
        OraDB.prepareCursor().close()



OraDB.createConn(OraDB.INFO) #con객체 생성
training_set_idx = np.random.permutation(7470)[:6000]
test_set_idx = np.random.permutation(7470)[6001:]


def clob2List(imageVal):
    return  list(map(int,imageVal.split(',')))


def classfiyDataSet():
    cur = OraDB.prepareCursor()
    xrayTabel = cur.execute('select * from xray_data')
    training_set_idx = np.random.permutation(7470)[:5999]
    test_set_idx = np.random.permutation(7470)[6000:]
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

for i in range(0,15):
    print(training_list[i][-1])


'''
코드작성 히스토리

목적 : 오라클로부터 데이터를 조회하여 훈련데이터와 검증데이터로 나눈 후 np.array타입으로 반환하고자 하는데 있다.
과정 : classfiyDataset 함수는 훈련데이터리스트와 검증데이터 리스트를 반환한다.
      1. table전체 조회를 실시한다.
      2. 튜플->리스트 로 변환한 각 row값을 각 변수로 할당한다. (현재는 idx가 의미가 없으므로 _ 처리)
      3. gray를 str->list형태로 바꿔주고 label을 맨 뒤 값으로 추가한다.
      4. gray값을 input_list에 append한다.
      5. 전체 input_list를 np.array화 한다. -> trsArray
      6. trsArray는 np.array이므로 np.random.permutation으로 설정한 idx를 적용한다.
      7. Conn객체를 종료한다.
      
결과 : 훈련데이터와 검증데이터가 분류되었다.

훈련데이터 shape =  (5999, 40001)
검증데이터 shape =  (1471, 40001)

'''

