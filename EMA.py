class HAdaptiveAvgPool2D(layers.Layer):
    def __init__(self, output_height):
        super(HAdaptiveAvgPool2D, self).__init__()
        self.output_height = output_height

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        h_factor = input_shape[1] // self.output_height
        x = tf.keras.backend.mean(
            inputs[:, :self.output_height * h_factor, :, :], axis=1, keepdims=True)
        return x

class WAdaptiveAvgPool2D(layers.Layer):
    def __init__(self, output_width):
        super(WAdaptiveAvgPool2D, self).__init__()
        self.output_width = output_width

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        w_factor = input_shape[2] // self.output_width
        x = tf.keras.backend.mean(
            inputs[:, :, :self.output_width * w_factor, :], axis=2, keepdims=True)
        return x
import tensorflow as tf
from tensorflow.keras import layers, Model

class EMA(tf.keras.Model):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = layers.Softmax(axis=-1)
        self.agp = layers.GlobalAveragePooling2D()
        self.pool_h = WAdaptiveAvgPool2D(1)
        self.pool_w = HAdaptiveAvgPool2D(1)
        self.gn = layers.GroupNormalization(groups=channels // self.groups, axis=-1)
        self.conv1x1 = layers.Conv2D(channels // self.groups, kernel_size=1, strides=1, padding='valid')
        self.conv3x3 = layers.Conv2D(channels // self.groups, kernel_size=3, strides=1, padding='same')

    def call(self, x):
        shape = tf.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        group_x = tf.reshape(x, (b * self.groups, h, w, c // self.groups))  # b*g,h,w,c//g
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])
        hw = self.conv1x1(tf.concat([x_h, x_w], axis=1))
        x_h, x_w = tf.split(hw, [h, w], axis=1)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])
        x1 = self.gn(group_x * tf.sigmoid(x_h) * tf.sigmoid(x_w))
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1))
        x11 =  tf.reshape(x11, (b * self.groups, 1, -1))

        print(x11.shape)
        x12 = tf.reshape(x2, (b * self.groups,c // self.groups, -1 ))  # b*g,c//g， hw, 
        x21 = self.softmax(self.agp(x2))
        x21 =  tf.reshape(x21, (b * self.groups, 1, -1))
        x22 = tf.reshape(x1, (b * self.groups,c // self.groups, -1 ))  # b*g,c//g， hw, 
        weights = tf.reshape(tf.matmul(x11, x12) + tf.matmul(x21, x22), (b * self.groups, h, w, 1))
        return tf.reshape(group_x * tf.sigmoid(weights), (b, h, w, c))

# Test the model with example input
input_shape = (None, 32, 32, 128)
model = EMA(128)
dummy_input = tf.random.normal((1, 32, 32, 128))  # 这里设置一个批次大小为1的示例输入
output = model(dummy_input)
print(output.shape)  # 输出形状应与输入形状相同
