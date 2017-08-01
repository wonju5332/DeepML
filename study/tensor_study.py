import tensorflow as tf

a = tf.constant([[3.,3.]])
b = tf.constant([[2.],[2.]])
product = tf.matmul(a,b)

print(product)  #Tensor("MatMul:0", shape=(1, 1), dtype=float32)

# default graph만 생성해 놓은 것.

# 행렬곱 값을 보고싶으면 세션에서 그래프를 실행해야 한다.

# sess = tf.Session()
#
# result = sess.run(product)
#
# print(result)  #[[ 12.]]


with tf.Session() as sess:
    result = sess.run([product])
    print(result)

