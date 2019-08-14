import tensorflow as tf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

# 학습을 위한 설정값들을 정의합니다.
learning_rate = .001
num_epochs = 30
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

# 입력값과 출력값을 받기 위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, output_size])

# ANN 모델을 정의합니다.
def build_ANN(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)
    W_output = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape=[output_size]))
    logits = tf.matmul(H2_output, W_output) + b_output

    return logits

#ANN 모델을 선언합니다.
predicted_value = build_ANN(x)

#손실 함수와 옵티마이저를 선언합니다.
# tf.nn.softmax_cross_entropt_with_logits 함수를 이용하여 활성 함수를 적용하지 않는 출력층의 결과값(logits)에 softmax 함수를 적용합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 세션을 열고 그래프를 실행합니다.
with tf.Session() as sess:
    # 변수들에 초기값을 할당합니다.
    sess.run(tf.global_variables_initializer())

    # epoch 만큼 최적화를 수행합니다.
    for epoch in range(num_epochs):
        average_loss = 0.
        # 전체 배치를 불러옵니다.
        total_batch = int(mnist.train.num_examples / batch_size)
        # 모든 배치들에 대해서 최적화를 수행합니다.
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # 옵티마이저를 실행해서 파라미터들을 업데이트 합니다.
            _, current_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
            # 평균 손실을 측정합니다.
            average_loss += current_loss / total_batch

    if epoch % display_step == 0:
        print("반복(Epoch): %d, 손실 함수(loss): %f" %((epoch + 1), average_loss))
        
    corrent_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
    print("정확도(accuracy): %f" %(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels})))
    
    