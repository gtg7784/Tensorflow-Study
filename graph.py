import tensorflow as tf

# 그래프 노드를 정의하고 출력합니다.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # 임시적으로 tf.float32 타입으로 선언될 것입니다.

print(node1, node2)

sess = tf.Session()  # 세션을 열고 그래프를 실행합니다.
print(sess.run([node1, node2]))

# 2개의 노드의 값을 더하는 연산을 수행하는 node3을 정의합니다.
node3 = tf.add(node1, node2)

print(node3)
print(sess.run(node3))

sess.close()