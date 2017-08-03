# 검색 키워드 = bulk insert / cx_oracle /

import cx_Oracle
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
portNum = 1522
SID = 'orcl3'

#######################################################

connection = cx_Oracle.connect('scott/tiger@localhost:1522/orcl3')


class Database():
    def __init__(self):
        self.ADMIN = None

    def createConn(self,info):
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


### Execute ###
db = Database()
db.createConn('scott/tiger@localhost:1522/orcl3')
emp_table = db.excuteSQL('select * from emp')  # 이터레이터 한 값을 전달한다. 튜플형태로 나와요.

for row in emp_table: #emp테이블이 출력된다.
    print(row)


### Fetch ###      #오직 조회쿼리에만 사용된다. 왜냐면 DDL, DCL은 결과를 반환하지 않기 때문입니다.

#cx_Oracle.Cursor.fetchall()
#cx_Oracle.Cursor.fetchmany([rows_no]
#cx_Oracle.Cursor.fetchone()


### CSV to Oracle ###

#
# ### Print db version ###
# vers = Database.createConn().version  #11.2.0.1.0


### create table ###

query = """
CREATE TABLE py_test (
column1 CLOB ,
label varchar2(50) )
"""
db.excuteSQL(query) # 생성


cursor = db.prepareCursor()

rows = [['asdasdssdasdaddvxcsd13','YES'],
        ['dfksdfkjdflskdfjfdkwww','NO']]
cursor.executemany('insert into py_test(column1, label) values (:1, :2)',rows)

for row in cursor.execute('select * from py_test'):
    print(row)


# 커밋한다.
db.dbCommit()





















