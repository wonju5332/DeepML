import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder("float",[None,784]) #todo 입력 노드가 784개이고 앞에 None 은 여기에 어떤 크기가 오던
                                       #todo 가능하다는 뜻이다. 학습과정에 사용될 이미지의 총개수가 될것이다.

W = tf.Variable(tf.zeros([784,10]))    #todo 출력층의 갯수가 10개여서 10으로 줌

# 텐써 이용하지 않았을때 ?
# self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)

b = tf.Variable(tf.zeros([10])) #todo 출력층의 노드의 갯수에 맞춰서 편향도 10개이다.
y = tf.nn.softmax(tf.matmul(x,W) + b) #todo affine 한 결과를 softmax 함수에 바로 입력해서 한번에 수행되고  #todo  예상값을 리턴하고 있다.
y_ = tf.placeholder("float",[None,10]) #todo 교차엔트로피를 구현하기 위해서 실제 레이블을 담고있는
                                        #todo 새로운 플레이스 홀더를 생성한다.



cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#todo 비용함수를 구현하는데 여기서 사용되는 reduce_sum 은
#todo 차원축소후 sum 하는 함수
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#todo 학습속도 0.01 과 SGD 경사하강법으로 비용함수의 오차가 
#todo 최소화 되겠금 역전파 시킴
sess = tf.Session()
#todo 텐써 플로우 그래프 연산을 시작하겠금 세션객체를 생성
sess.run(tf.global_variables_initializer())
#todo 모든 변수를 초기화 한다.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #todo 훈련 데이터셋에서 무작위로 100개를 추출

    print(sess.run(y, feed_dict={x:[batch_xs]}))

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #todo 100개의 데이터를 SGD 의 경사감소법으로 훈련시킨다.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #todo y 라벨(예상)중 가장 큰 인덱스를 리턴하고 
    #todo y_라벨(실제값) 중 가장 큰 인덱스를 리턴해서 같은지 비교
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    # todo [True,False,True,True] 를 [1,0,0,1] 로 변경 될것이고 
    # todo 평균 0.75 가 출력된다.ㅏ
    print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    # todo 정확도가 출력이 된다.
    