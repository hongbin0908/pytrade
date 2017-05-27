import numpy as np
import tensorflow as tf
import math

class Model:
    def fit(self, x_data, y_data):
        pass
    def predict(self, x_data):
        pass


class ModelMdn(Model):
    def __init__(self, inputsize = 1,  hidden_size = 60, model_size = 60, lr = 0.0001):
        NHIDDEN = hidden_size
        STDEV = 0.01
        self.KMIX = model_size  # number of mixtures
        NOUT = self.KMIX * 3  # pi, mu, stdev
        self.x = tf.placeholder(dtype=tf.float64, shape=[None, inputsize], name="x")
        self.y = tf.placeholder(dtype=tf.float64, shape=[None, 1], name="y")
        Wh = tf.Variable(tf.random_normal([inputsize, NHIDDEN], stddev=STDEV, dtype=tf.float64))
        #Wh = tf.Variable(tf.zeros([inputsize, NHIDDEN], dtype=tf.float64))
        #bh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))
        bh = tf.Variable(tf.zeros ([1, NHIDDEN], dtype=tf.float64))
        Wh1 = tf.Variable(tf.random_normal([NHIDDEN, NHIDDEN], stddev=STDEV, dtype=tf.float64))
        bh1 = tf.Variable(tf.zeros ([1, NHIDDEN], dtype=tf.float64))
        Wo = tf.Variable(tf.random_normal([NHIDDEN, NOUT], stddev=STDEV, dtype=tf.float64))
       # bo = tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))
        #Wo = tf.Variable(tf.zeros([NHIDDEN, NOUT],dtype=tf.float32))
        bo = tf.Variable(tf.zeros([1, NOUT], dtype=tf.float64))
        hidden_layer = tf.nn.relu(tf.matmul(self.x, Wh) + bh)
        hidden_layer1 = tf.nn.relu(tf.matmul(hidden_layer, Wh1) + bh1)
        self.output = tf.matmul(hidden_layer, Wo) + bo
        out_pi, out_sigma, out_mu = self._get_mixture_coef(self.output)
        self.lossfunc = self._get_lossfunc(out_pi, out_sigma, out_mu, self.y)
        self.train_op = tf.train.AdamOptimizer(learning_rate= lr, beta1=0.0001).minimize(self.lossfunc)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def fit(self, x_, y_):
        y_ = np.reshape(y_, (-1, 1))
        self.NEPOCH = 300
        self.loss = np.zeros(self.NEPOCH)  # store the training progress here.

        for i in range(self.NEPOCH):
            self.sess.run(self.train_op, feed_dict={self.x: x_, self.y: y_})
            loss = self.sess.run(self.lossfunc, feed_dict={self.x: x_, self.y: y_})
            print("step %d: loss=%.4f" %(i, loss) )
            self.loss[i] = loss


    def predict_distribution(self, x_):
        out_pi, out_sigma, out_mu = self.sess.run(self._get_mixture_coef(self.output),feed_dict={self.x: x_})
        return self._generate_ensemble(out_pi, out_mu, out_sigma)




    def predict_value(self, x_, is_debug = False):
        out_pi, out_sigma, out_mu = self.sess.run(self._get_mixture_coef(self.output), feed_dict={self.x:x_})

        print(out_pi)
        return self.get_maxprob_mu(out_pi, out_sigma, out_mu, x_.shape[1], is_debug)

    def get_maxprob_mu(self, out_pi, out_sigma, out_mu, x_shape, is_debug = False):
        debug_result = []
        SAMPLE_NUM = len(out_pi)
        result = np.zeros((SAMPLE_NUM, 1), dtype=float)
        debug_result = np.zeros((SAMPLE_NUM, 1), dtype = float)
        tmp_result1 = np.zeros((SAMPLE_NUM, 1), dtype = float)
        tmp_result2 = np.zeros((SAMPLE_NUM, 1), dtype = float)
        for i in range(0, SAMPLE_NUM):
            max_value = -10000
            max_index = -1

            for j in range(0, self.KMIX):
                tmp = out_pi[i][j]
                if out_mu[i][j] > 1:
                    tmp_result1[i] += out_pi[i][j] * 1.0/out_sigma[i][j]
                else:
                    tmp_result2[i] += out_pi[i][j] * 1.0 / out_sigma[i][j]
                if max_value < tmp:
                    max_value = tmp
                    max_index = j
                #result1 = self._tf_normal(1.0, out_mu[i][j], out_sigma[i][j])
            #result[i] = result[i]/self.KMIX
            if is_debug == True:
                tmp_result = np.abs(tmp_result1[i] - tmp_result2[i])
                debug_result[i] =  out_pi[i][max_index] * 1.0 / out_sigma[i][max_index]
            result[i] = out_mu[i][max_index]
        return result, debug_result

    def _get_mixture_coef(self, output):
        out_pi = tf.placeholder(dtype=tf.float64, shape=[None, self.KMIX], name="mixparam")
        out_sigma = tf.placeholder(dtype=tf.float64, shape=[None, self.KMIX], name="mixparam")
        out_mu = tf.placeholder(dtype=tf.float64, shape=[None, self.KMIX], name="mixparam")
        out_pi, out_sigma, out_mu = tf.split(output, 3, 1)
        max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
        out_pi = tf.subtract(out_pi, max_pi)
        out_pi = tf.exp(out_pi)
        normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
        out_pi = tf.multiply(normalize_pi, out_pi)
        out_sigma = tf.exp(out_sigma)
        tmp1 = tf.constant(0.0001, dtype=tf.float64)
        out_sigma = tf.maximum(tmp1, out_sigma)
        return out_pi, out_sigma, out_mu

    def _tf_normal(self, y, mu, sigma):
        oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)  # normalisation factor for gaussian, not needed.
        result = tf.subtract(y, mu)
        result = tf.multiply(result, tf.reciprocal(sigma))
        result = -tf.square(result) / 2
        return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * oneDivSqrtTwoPI

    def _get_lossfunc(self, out_pi, out_sigma, out_mu, y):
        tmp1 = tf.constant(0.0001, dtype=tf.float64)
        out_sigma = tf.maximum(tmp1, out_sigma)
        result = self._tf_normal(y, out_mu, out_sigma)
        tmp = tf.constant(1.0, dtype = tf.float64)
        result = tf.minimum(tmp, result)
        result = tf.multiply(result, out_pi)


        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result)
        result = tf.reduce_mean(result)

        result2 = tf.reduce_mean(tf.multiply(out_mu, out_mu))



        return result + result2
    def _get_pi_idx(self, x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print('error with sampling ensemble')
        return -1

    def _generate_ensemble(self, out_pi, out_mu, out_sigma, M = 10):
        NTEST = len(out_mu)
        result = np.random.rand(NTEST, M) # initially random [0, 1]
        rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
        mu = 0
        std = 0
        # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                idx = self._get_pi_idx(result[i, j], out_pi[i])
                mu = out_mu[i, idx]
                std = out_sigma[i, idx]
                result[i, j] = mu + rn[i, j]*std
        return result