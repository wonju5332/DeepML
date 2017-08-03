# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# 드롭아웃 사용 유무와 비울 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False

# 수치만큼 노드 제외
dropout_ratio_1 = 0.2
dropout_ratio_2 = 0.5
dropout_ratio_3 = 0

# ====================================================
network = []
trainer = []
train_acc_list = []
test_acc_list = []

# # 0.2 dropout
# net1 = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
#                               output_size=10, use_dropout=True, dropout_ration=dropout_ratio_1)
#
# network.append(net1)
# trainer.append(Trainer(net1, x_train, t_train, x_test, t_test,
#                   epochs=301, mini_batch_size=100,
#                   optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True))

# # 0.5 dropout
# net2 = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
#                               output_size=10, use_dropout=True, dropout_ration=dropout_ratio_2)
#
# network.append(net2)
# trainer.append(Trainer(net2, x_train, t_train, x_test, t_test,
#                   epochs=301, mini_batch_size=100,
#                   optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True))

# not dropout
net3 = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=False, dropout_ration=dropout_ratio_3)
network.append(net3)
trainer.append(Trainer(net3, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True))

for t in trainer:
    t.train()
    train_acc_list.append(t.train_acc_list)
    test_acc_list.append(t.test_acc_list)


# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}

for i in range(len(train_acc_list)):
    x = np.arange(len(train_acc_list[i]))
    plt.plot(x, train_acc_list[i], marker='o', label='train' + str(i), markevery=10)
    plt.plot(x, test_acc_list[i], marker='s', label='test' + str(i), markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')

plt.show()