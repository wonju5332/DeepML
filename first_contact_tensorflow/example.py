import numpy as np
import tensorflow as tf


x = np.arange(6).reshape(2,3)
print(x)


sess = tf.Session()
print(sess.run(tf.reduce_sum(x)))  # 2,2의 행렬을 1차원으로 축소하여 sum했다.

print(sess.run(tf.reduce_sum(x,0)))  # [3 5 7]  # 0:열단위 1:행단위  로 sum한다.
print(sess.run(tf.reduce_sum(x,1)))  # [3 12]






x = tf.zeros([2,3]) #0으로 2,3 생성
y = tf.ones([2,3])  #1로 2,3 생성


result = tf.add(x,y) #둘을 더한다.

sess = tf.Session() #todo 이 코드 전까지는, 아무것도 실행되지 않고 get-set-ready 상태.

print(sess.run(x)) #todo 실행한 결과   #  start!
print(sess.run(y))
print(sess.run(result))




x = tf.placeholder("float", [2,3])
y = tf.placeholder("float",[2,3])
result = tf.add(x,y)

sess = tf.Session()
a = sess.run(result, feed_dict={x:[[2,2,2],[2,2,2]], y:[[3,3,3],[3,3,3]]})

print(a)






x = tf.placeholder("float", [2,3])
w = tf.placeholder("float", [3,2])
result = tf.matmul(x,w)

sess = tf.Session()

print(sess.run(result,feed_dict={x:[[2,2,2],[2,2,2]],w:[[3,3],[3,3],[3,3]]}))




correct_prediction = [ True, False , True  ,True  ,True  ,True  ,True,  True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True, False , True  ,True, False , True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True,
  True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True ,False , True  ,True  ,True  ,True  ,True
  ,True  ,True, False , True, False , True  ,True  ,True  ,True  ,True  ,True  ,True
  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True  ,True
 ,False , True  ,True  ,True ]


sess = tf.Session()

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy))





