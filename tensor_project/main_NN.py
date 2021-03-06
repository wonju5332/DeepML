# from Project_Git.DeepLearningProject.p02_cnn_image_classification.xrayss.model import Model
import tensorflow as tf
import numpy as np
import time
from tensor_project.xray_image_to_list_junho_ver import OraDB
import cx_Oracle

# training_epochs = 20
batch_size = 200

training_set_idx = np.random.permutation(7470)[:6000]
test_set_idx = np.random.permutation(7470)[6001:]


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

    input_list = []
    for row in xrayTabel:
        _, gray, label = list(row)
        gray = cx_Oracle.LOB.read(gray)  # str형태
        gray = clob2List(gray)  # [255,322,] list형태
        input_list.append(gray+[int(label)])  # list형태로 넣는다. [list, int]
    trsArray = np.array(input_list)  # array( [ [],[] ] )
    training_list = trsArray[training_set_idx]
    test_list = trsArray[test_set_idx]
    print('훈련데이터 shape = ', training_list.shape)
    print('검증데이터 shape = ', test_list.shape)

    OraDB.releaseConn() #con객체를 닫아준다.

    return training_list, test_list




########################################################################################################################
## ▣ Data Training
##  - train data : 50,000 개 (10클래스, 클래스별 5,000개)
##  - epoch : 20, batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    # 시작 시간 체크
    stime = time.time()
    models = []
    num_models = 5
    train_file_list, test_file_list = classfiyDataSet()

    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Learning Started!')

    early_stopping_list = []
    last_epoch = -1
    epoch = 0
    early_stop_count = 0

    #여기서 execute(전체 조회)


    while True:
        sstime = time.time()
        avg_cost_list = np.zeros(len(models))
        for index in range(0, len(train_file_list)):
            total_x, total_y = data_setting(train_file_list[index])
            for start_idx in range(0, 1000, batch_size):
                train_x_batch, train_y_batch = total_x[start_idx:start_idx+batch_size], total_y[start_idx:start_idx+batch_size]
                # print(len(train_x_batch[0]))
                # print(len(train_y_batch[0]))

                for idx, m in enumerate(models):
                    c, _ = m.train(train_x_batch, train_y_batch)
                    avg_cost_list[idx] += c / batch_size

        ################################################################################################################
        ## ▣ early stopping - Created by 배준호
        ##  - prev epoch 과 curr epoch 의 cost 를 비교해서 curr epoch 의 cost 가 더 큰 경우 종료하는 기능
        ################################################################################################################
        saver.save(sess, 'log/epoch_' + str(epoch + 1) +'.ckpt')
        early_stopping_list.append(avg_cost_list)
        diff = 0
        if len(early_stopping_list) >= 2:
            temp = np.array(early_stopping_list)
            last_epoch = epoch
            diff = np.sum(temp[0] < temp[1])
            if diff > 2:
                early_stop_count += 1
                print('----------------------||   Early Stopped   ||----------------------')
                print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list, ' - ', diff)
                print('early stopping - epoch({})'.format(epoch + 1), ' early stopped ', early_stop_count + 1)
                print('---------------------------------------------------------------------')
                if early_stop_count == 3:
                   break
            early_stopping_list.pop(0)
        epoch += 1
        eetime = time.time()
        print('Epoch: ', '%04d' % (epoch), 'cost =', avg_cost_list, ' - ', diff, ', epoch{} time'.format(epoch), round(eetime - sstime, 2))

    print('Learning Finished!')

    # 종료 시간 체크
    etime = time.time()
    print('consumption time : ', round(etime-stime, 6))

tf.reset_default_graph()

########################################################################################################################
## ▣ Data Test
##  - test data : 10,000 개
##  - batch_size : 100, model : 5개
########################################################################################################################
with tf.Session() as sess:
    models = []
    num_models = 5
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, 'log/epoch_' + str(epoch) + '.ckpt')

    print('Testing Started!')

    ensemble_accuracy = 0.
    model_accuracy = [0., 0., 0., 0., 0.]
    cnt = 0
    ensemble_confusion_mat = np.zeros((10, 10))

    for index in range(0, len(test_file_list)):
        total_x, total_y = data_setting(test_file_list[index])
        for start_idx in range(0, 1000, batch_size):
            test_x_batch, test_y_batch = total_x[start_idx:start_idx + batch_size], total_y[start_idx:start_idx + batch_size]
            test_size = len(test_y_batch)
            predictions = np.zeros(test_size * 10).reshape(test_size, 10)

            model_result = np.zeros(test_size*10, dtype=np.int).reshape(test_size, 10)
            model_result[:, 0] = range(0, test_size)

            for idx, m in enumerate(models):
                model_accuracy[idx] += m.get_accuracy(test_x_batch, test_y_batch)
                p = m.predict(test_x_batch)
                model_result[:, 1] = np.argmax(p, 1)
                for result in model_result:
                    predictions[result[0], result[1]] += 1

            ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y_batch, 1))
            ensemble_accuracy += tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
            ensemble_confusion_mat = tf.add(tf.contrib.metrics.confusion_matrix(labels=tf.argmax(test_y_batch, 1), predictions=tf.argmax(predictions, 1), num_classes=10, dtype='int32', name='confusion_matrix'), ensemble_confusion_mat)
            cnt += 1
    for i in range(len(model_accuracy)):
        print('Model ' + str(i) + ' : ', model_accuracy[i] / cnt)
    print('Ensemble Accuracy : ', sess.run(ensemble_accuracy) / cnt)
    print('Testing Finished!')
    print('####### Confusion Matrix #######')
    # print(sess.run(ensemble_confusion_mat))
    # print(sess.run(tf.contrib.metrics.confusion_matrix(labels=tf.arg_max(total_y, dimension=1), predictions=tf.arg_max(m.predict(total_x), dimension=1), num_classes=10, dtype='int32', name='confusion_matrix')))