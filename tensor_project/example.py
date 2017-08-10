import tensorflow as tf





#########


a = tf.placeholder("float") # 공간을 만든다.
b = tf.placeholder("float")
y = tf.multiply(a,b)  #곱을 한다.
with tf.Session() as sess:
    sess.run(y, feed_dict={a:10,b:32})



a = tf.constant(10)
b = tf.constant(32)
result = tf.multiply(a, b)

with tf.Session() as sess:
    sess.run(result)



x = tf.constant([[3.0,3.0],
                 [5.0,5.0]])

w = tf.constant([[2.0,2.0],
                 [3.0,3.0]])

b = tf.constant([[1.0,1.0]])

product = tf.matmul(x,w)

with tf.Session() as sess:
    y = sess.run(product+b)
    print(y)





# # 변수를 0으로 초기화
# state = tf.Variable(0, name="counter")
#
# # state에 1을 더할 오퍼레이션 생성  --> 그래프 상의 계산노드에 해당 : 계산된 텐서를 반환한다.
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# # 그래프는 처음에 변수를 초기화해야 합니다. 아래 함수를 통해 init 오퍼레이션을 만듭니다.
# init_op = tf.initialize_all_variables()
#
# # 그래프를 띄우고 오퍼레이션들을 실행
# with tf.Session() as sess:
#   # 초기화 오퍼레이션 실행
#   sess.run(init_op)
#   # state의 초기 값을 출력
#   print(sess.run(state))
#   # state를 갱신하는 오퍼레이션을 실행하고, state를 출력
#   for _ in range(3):
#     sess.run(update)
#     print(sess.run(state))\