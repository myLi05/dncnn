from PIL import Image
import scipy.misc as misc
import os
import tensorflow as tf
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import numpy as np

H = 256
W = 256
C = 1
DEPTH = 20
beta1   = 0.9
beta2   = 0.999
batch_size = 1
EPOCHS = 50
SIGMA = 25
epsilon = 1e-8


def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn, reuse=tf.AUTO_REUSE):
        beta = tf.get_variable(name='beta', shape=[x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def InstanceNorm(inputs, name):
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        scale = tf.get_variable("scale", shape=mean.shape[-1], initializer=tf.constant_initializer([1.]))
        shift = tf.get_variable("shift", shape=mean.shape[-1], initializer=tf.constant_initializer([0.]))
        return (inputs - mean) * scale / tf.sqrt(var + 1e-10) + shift

def conv(name, inputs, nums_out, ksize, strides, padding="SAME", is_SN=False):
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[ksize, ksize, int(inputs.shape[-1]), nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", shape=[nums_out], initializer=tf.constant_initializer(0.))
        return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding) + b

def deconv(name, inputs, nums_out, ksize, strides, padding="SAME"):
    with tf.variable_scope(name):
        w = tf.get_variable("W", shape=[ksize, ksize, nums_out, int(inputs.shape[-1])], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer(0.))
        # inputs = tf.image.resize_nearest_neighbor(inputs, [H*strides, W*strides])
        # return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], padding) + b
    return tf.nn.conv2d_transpose(inputs, w, [tf.shape(inputs)[0], int(inputs.shape[1])*strides, int(inputs.shape[2])*strides, nums_out], [1, strides, strides, 1], padding=padding) + b


def fully_connected(name, inputs, nums_out):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", [int(inputs.shape[-1]), nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer(0.))
        return tf.matmul(inputs, W) + b



def leaky_relu(x, slope=0.2):
    return tf.maximum(x, slope*x)


class net:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = tf.nn.relu(conv("conv0", inputs, 64, 3, 1))
            for d in np.arange(1, DEPTH - 1):
                inputs = tf.nn.relu(batchnorm(conv("conv_" + str(d + 1), inputs, 64, 3, 1), train_phase, "bn" + str(d)))
            inputs = conv("conv" + str(DEPTH - 1), inputs, C, 3, 1)
            return inputs

class DnCNN:
    def __init__(self):
        self.clean_img = tf.placeholder(tf.float32, [batch_size, H, W, C])
        self.noised_img = tf.placeholder(tf.float32, [batch_size, None, None, C])
        self.train_phase = tf.placeholder(tf.bool)
        dncnn = net("DnCNN")
        self.res = dncnn(self.noised_img, self.train_phase)
        self.denoised_img = self.noised_img - self.res
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.res - (self.noised_img - self.clean_img)), [1, 2, 3]))
        self.Opt = tf.train.AdamOptimizer(0.01,beta1= beta1 ,beta2= beta2 ).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        filepath = "C:/Users/limengyang/Desktop/cleandenoise//"
        cleandata = os.listdir(filepath)
        noisedata = os.listdir('C:/Users/limengyang/Desktop/noiseresult2')
        saver = tf.train.Saver()
        for epoch in range(50):
            for i in range(cleandata.__len__()//batch_size ):
                noise_image = np.zeros([batch_size, H, W, C])
                clean_image = np.zeros([batch_size, H, W, C])
                for idx,filename in enumerate(noisedata[i*batch_size :i * batch_size +  batch_size ]):
                    noise_image[idx,:,:,0] = np.array(Image.open('C:/Users/limengyang/Desktop/noiseresult2//' + filename))
                for idx1,filename1 in enumerate(cleandata[i*batch_size :i * batch_size +  batch_size]):
                    clean_image[idx1,:,:,0] = np.array(Image.open('C:/Users/limengyang/Desktop/cleandenoise//' + filename1))
                #noised_batch = cleaned_batch + np.random.normal(0, SIGMA, cleaned_batch.shape)
                self.sess.run(self.Opt, feed_dict={self.clean_img: clean_image, self.noised_img: noise_image, self.train_phase: True})
                if i % 10 == 0:
                    [loss, denoised_img] = self.sess.run([self.loss, self.denoised_img], feed_dict={self.clean_img: clean_image, self.noised_img: noise_image, self.train_phase: False})
                    print("Epoch: %d, Step: %d, Loss: %g"%(epoch, i, loss))
                    compared = np.concatenate((clean_image[0, :, :, 0], noise_image[0, :, :, 0], denoised_img[0, :, :, 0]), 1)
                    Image.fromarray(np.uint8(compared)).save("./dncnnresult//"+str(epoch)+"_"+str(i)+".jpg")
                if i % 500 == 0:
                    saver.save(self.sess, "./dncnnpara//DnCNN.ckpt")
    def test(self):
        saver = tf.train.Saver()
        psnrnum = 0
        ssimnum = 0
        saver.restore(self.sess, 'E:/pythonproject/noiseexperiment/dncnnpara/DnCNN.ckpt')
        X = os.listdir('C:/Users/limengyang/Desktop/VAdenoise')
        Z = os.listdir('C:/Users/limengyang/Desktop/VAnoise')
        #Cor = os.listdir('C:/Users/limengyang/Desktop/denoisemohu0')
        for i in range(Z.__len__()):
            #x = np.array(Image.open('C:/Users/limengyang/Desktop/sizenoise0/' + image0[i]))[np.newaxis, :, :,np.newaxis] / 127.5 - 1.0
            x = np.reshape(np.array(Image.open('C:/Users/limengyang/Desktop/VAdenoise//' + X[i])), [1, 256, 256, 1])
            z = np.reshape(np.array(Image.open('C:/Users/limengyang/Desktop/VAnoise//' + Z[i])), [1, 256, 256, 1])
            #z1 = np.reshape(z,[1,z.shape[0],z.shape[1],1])
            image = self.sess.run(self.denoised_img, feed_dict={self.noised_img: z, self.train_phase: False})
            psnrnum += psnr(np.uint8(x[0, :, :, 0]), np.uint8(image[0, :, :, 0]))
            ssimnum += ssim(np.uint8(x[0, :, :, 0]), np.uint8(image[0, :, :, 0]))
            #psnrnum.append(psnr(np.uint8(x[0,:,:,0]),np.uint8(image[0,:,:,0])))
            #ssimnum.append(ssim(np.uint8(x[0,:,:,0]),np.uint8(image[0,:,:,0])))
            #image1 = np.concatenate((mapping(z[0, :, :, 0]), mapping(image[0, :, :, 0])), axis=1)
            #Image.fromarray(np.uint8(image)).save('C:/Users/limengyang/Desktop/result2//'+str(Z[i])+'.jpg')
        PSNR0 = psnrnum/150
        SSIM0 = ssimnum/150
        print(PSNR0)
        print(SSIM0)
    # def test(self):
    #     saver = tf.train.Saver()
    #     saver.restore(self.sess, "./compare/dncnnpara/DnCNN.ckpt")
    #     # X = os.listdir('C:/Users/limengyang/Desktop/cleandenoise')
    #     Z = os.listdir('./test/binanoise1')
    #     # Cor = os.listdir('C:/Users/limengyang/Desktop/denoisemohu0')
    #     for i in range(Z.__len__()):
    #         # x = np.array(Image.open('C:/Users/limengyang/Desktop/sizenoise0/' + image0[i]))[np.newaxis, :, :,np.newaxis] / 127.5 - 1.0
    #         z = np.array(Image.open('./test/binanoise1//' + Z[i]))
    #         noised_img = np.reshape(z, [1, z.shape[0], z.shape[1], 1])
    #         image = self.sess.run(self.denoised_img, feed_dict={self.noised_img: noised_img, self.train_phase: False})
    #         Image.fromarray(np.uint8(image[0, :, :, 0])).save('./compar/result//' + str(Z[i]) + '.jpg')


if __name__ == "__main__":
    dncnn = DnCNN()
    dncnn.test()

