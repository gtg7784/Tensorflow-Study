import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Mnist 데이터를 다운로드 합니다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=False)  # one_hot = Flase 로 불러옵니다.

# 입력값과 출력값을 받기 위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.int64, shape=[None])  # tf.int64 타입으로 선언합니다.

# 변수들을 설정하고 소프트맥스 회귀 모델을 정의합니다.
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
logits = tf.matmul(x, W) + b
y_pred = tf.nn.softmax(logits)

# cross-entropy 손실 함수와 옵티마이저를 정의합니다.
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
# tf.nn.softmax_cross_entropy_with_logits API 를 이용한 구현
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))

# tf.nn.spare_softmax_cross_entropy_with_logits API 를 이용한 구현
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 세션을 열고 변수들에 초기값을 할당합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 1000 번 반복을 수행하면서 최적화를 수행합니다.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

#학습이 끝나면 학습된 모델의 정확도를 출력합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("정확도(Accuracy): %f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

sess.close()