import tensorflow as tf
with tf.variable_scope("foo"): #create the first time
    v = tf.get_variable("v", [1])

with tf.variable_scope("foo", reuse=True): #reuse the second time
    v = tf.get_variable("v", [1])

with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

# 下面是在另外一个命名空间来定义变量的
with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

# 所以，实际上weights1 和 weights2 这两个引用名指向了不同的空间，不会冲突
print(weights1.name)
print(weights2.name)
# 注意，这里的 with 和 python 中其他的 with 是不一样的
# 执行完 with 里边的语句之后，这个 conv1/ 和 conv2/ 空间还是在内存中的。这时候如果再次执行上面的代码
# 就会再生成其他命名空间



# 这里是正确的打开方式~~~可以看出，name 参数才是对象的唯一标识
with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.get_variable('bias', shape=[3])

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')

print(Weights1.name)
print(Weights2.name)
# 可以看到这两个引用名称指向的是同一个内存对象