import tensorflow as tf
x=tf.Variable([1,2])
a=tf.constant([3,3])
sub=tf.subtract(x,a)
add=tf.add(x,sub)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

state=tf.Variable(0,name="counter")
new_state=tf.add(state,1)
update=tf.assign(state,new_state)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
    
    
