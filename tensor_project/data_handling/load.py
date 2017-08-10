import cx_Oracle
import numpy.random as rnd
import numpy as np
import time
# import os
#
# cursor = connection.cursor()
# aa = ['WONJU',300]
# cursor.executemany('insert into emp(ename,sal) values (:1, :2)', )
#
#
# for i in cursor.execute('select * from emp'):
#     print(i)



#######################################################
# 접속 DB  정보 #########################################
#######################################################
UserID = 'scott'
PassWd = 'tiger'
portNum = 1521
SID = 'orcl'




class Database():
    def __init__(self):
        self.ADMIN = None

    def createConn(self, info):
        self.ADMIN = cx_Oracle.connect(info)
        print('DB 커넥션 객체가 생성되었습니다.')

    def prepareCursor(self):
        return self.ADMIN.cursor()

    def excuteSQL(self, query):
        cur = self.prepareCursor()

        return cur.execute(query)

    def dbCommit(self):
        return self.ADMIN.commit()


#################### 수행절 ###############

### Oracle to Python###

stime = time.time()

### Execute ###
db = Database()
db.createConn('scott/tiger@localhost:1522/orcl3')
sqlquery = 'select * from xray_data'
sqlquery_excute = db.excuteSQL(sqlquery)  # 이터레이터 한 값을 전달한다. 튜플형태로 나와요.
print('쿼리문 ',sqlquery,'; 가 실행되었습니다.')

xrayTabel = db.excuteSQL('select * from xray_data')

tempList = []


def clob2List(imageVal):
    return list(map(int,imageVal.split(',')))


for row in xrayTabel:
    idx, gray, label = list(row)
    gray = cx_Oracle.LOB.read(gray)
    gray = clob2List(gray)
    tempList.append([idx, gray, label])

# dataset = []
#
# for row in sqlquery_excute:  # emp테이블이 출력된다.
#     eachImage = list(row)  #튜플 -> 리스트로 변환
#     # dataset.append([a[0], cx_Oracle.LOB.read(a[1]), a[2]])
#     dataset.append([a[0], cx_Oracle.LOB.read(a[1]), a[2]])


new_dataset = []
new2_dataset = []
for data in tempList:  #dataset
    new_dataset.append([int(data[0]), [int(i) for i in data[1].split(",")], [int(data[2])]])
  #new_dataset = [idx, list, label]


print(new2_dataset[0])
#
new_array = []
for array in new_dataset:
    new_array.append(array[1]+array[2])

new_array = np.array(new_array)

ftime = time.time()







print(new_array)
print(type(new_array))
print(new_array.shape)
print('Run time : ', ftime-stime)



# training_set_idx = rnd.permutation(7470)[:6000]
# test_set_idx = rnd.permutation(7470)[6001:]

