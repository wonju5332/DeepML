import tensorflow as tf
import numpy as np
from mnist import load_mnist

##### mnist 데이터 불러오기 및 정제 #####

############################################
# mnist 데이터 중 10000개 저장
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True)
# input = np.concatenate((x_train, x_test), axis=0)
# target = np.concatenate((t_train, t_test), axis=0)
# print('input shape :', input.shape, '| target shape :', target.shape)
# a = np.concatenate((input, target), axis=1)
# np.savetxt('mnist.csv', a[:10000], delimiter=',')
# print('mnist.csv saved')    #array를 csv로 저장한다. (한번 하고나면 파일 생기니까 주석처리)
############################################

# 파일 로드 및 변수 설정
save_status = True
load_status = True


mnist = np.loadtxt('mnist.csv', delimiter=',', unpack=False, dtype='float32')
print('mnist.csv loaded')
print('mnist shape :',mnist.shape)

train_num = int(mnist.shape[0] * 0.8)  #8000개

x_train, x_test = mnist[:train_num,:784], mnist[train_num:,:784]
t_train, t_test = mnist[:train_num,784:], mnist[train_num:,784:]

print('x train shape :',x_train.shape, '| x target shape :',x_test.shape)
print('t train shape :',t_train.shape, '| t target shape :',t_test.shape)


global_step = tf.Variable(0, trainable=False, name='global_step')
X = tf.placeholder(tf.float32,[None, 784])
T = tf.placeholder(tf.float32,[None, 10])
W = tf.Variable(tf.random_uniform([784,10], -1e-7, 1e-7)) # [784,10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 어레이
b = tf.Variable(tf.random_uniform([10], -1e-7, 1e-7))    # [10] 형상을 가진 -1e-7 ~ 1e-7 사이의 균등분포 벡터
Y = tf.add(tf.matmul(X,W), b) # tf.matmul(X,W) + b 와 동일

############################################
# 그외 가중치 초기화 방법
# W = tf.Variable(tf.random_uniform([784,10], -1, 1)) # [784,10] 형상을 가진 -1~1 사이의 균등분포 어레이
# W = tf.get_variable(name="W", shape=[784, 10], initializer=tf.contrib.layers.xavier_initializer()) # xavier 초기값
# W = tf.get_variable(name='W', shape=[784, 10], initializer=tf.contrib.layers.variance_scaling_initializer()) # he 초기값
# b = tf.Variable(tf.zeros([10]))
############################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=T, logits=Y)) # 한번에 soft-cross-sum 한번에
optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(cost, global_step=global_step)
#글로벌 스텝은 옵티마이저를 돌때마다, 1개씩 증가한다. 즉 Epoch이 얼마나 쌓이는지 보는 것이다.

############################################
# 그외 옵티마이저
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01)
############################################

##### mnist 학습시키기 #####
# 일반 버전
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

##위 방법도 있지만, 아래방법을 보는게 더 중요하다.

#todo 로드 버전


sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())  #저장

cp = tf.train.get_checkpoint_state('./save') # save 폴더를 checkpoint로 설정
# checkpoint가 설정되고, 폴더가 실제로 존재하는 경우 restore 메소드로 변수, 학습 정보 불러오기
if cp and tf.train.checkpoint_exists(cp.model_checkpoint_path):  #save라는 폴더가 있다면.
    saver.restore(sess, cp.model_checkpoint_path)    #변수들을 회복시켜라, 저장되어 있는 변수를 로드하라.
    print(sess.run(global_step),'회 학습한 데이터 로드 완료')
# 그렇지 않은 경우 일반적인 sess.run()으로 tensorflow 실행
else:
    sess.run(tf.global_variables_initializer())
    print('새로운 학습 시작')

#todo                <epoch, batch 설정>

epoch = 100  # 8000개의 데이터를 100개씩 배치로 하면 80번을 도는데 이 80번 도는것을 1에폭이라 한다.
total_size = x_train.shape[0]
batch_size = 100
# mini_batch_size = 100
total_batch = int(total_size/batch_size)

# 정확도 계산 함수
correct_prediction = tf.equal(tf.argmax(T, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 설정한 epoch 만큼 루프
for each_epoch in range(epoch):
    total_cost = 0
    # 각 epoch 마다 batch 크기만큼 데이터를 뽑아서 학습
    # for idx in range(0, total_size, batch_size):
    for idx in range(0, 8000, 100):
        batch_x, batch_y = x_train[idx:idx+batch_size], t_train[idx:idx+batch_size]
                                  #0:100 .. 100:200 .. 500:600...
        _, cost_val = sess.run([optimizer, cost], feed_dict={X : batch_x, T : batch_y})
        total_cost += cost_val #cost값을 계속 쌓는다.

    print('Epoch:', '%04d' % (each_epoch + 1),
          'Avg. cost =', '{:.8f}'.format(total_cost / total_batch),
          )

print('최적화 완료!')

#todo 최적화가 끝난 뒤, 변수와 학습 정보 저장
# saver.save(sess, './save/mnist_dnn.ckpt', global_step=global_step)

##### 학습 결과 확인 #####
print('Train 정확도 :', sess.run(accuracy, feed_dict={X: x_train, T: t_train}))
print('Test 정확도:', sess.run(accuracy, feed_dict={X: x_test, T: t_test}))



'''

# 문제1. 1) 가중치 초기값을 xavier 초기값으로 설정, 2) 옵티마이저를 momentum 옵티마이저로 설정 후, 3) epoch은 200번,
#       4) batch_size 는 200으로 수정하여 학습해보기

# 문제2. 19~20번째 줄의 save_status 와 load_status가 각각 True 인 경우에만 저장/불러오기 되도록 코드 수정

# 문제3. 82번째 줄의 mini_batch_size를 이용하여 200개의 배치 데이터 중 100개만 랜덤으로 뽑아 학습하도록 코드 수정
#       (힌트 : np.random.randint(low=a, high=b, size=c) --> 숫자 a~b 사이의 정수 c개를 랜덤으로 뽑아주는 함수)

# 문제4. 훈련데이터의 10%를 뽑아 만든 검증 데이터로 아래 형식과 같이 50번째 epoch 마다 정확도 출력해보기.
#         (훈련데이터(0.9) + 검증데이터(0.1) = 전체의 80%   /   테스트 데이터 = 전체의 20%)

'''