# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from glob import glob
import time


from ops import *
from utils import *

class Multigrid(object):
    def __init__(self, sess, flags, scale_list = [1, 4, 16, 64]):
        """
        """
        self.sess = sess
        self.batch_size = flags.batch_size
        self.weight_decay = flags.weight_decay
        self.scale_list = scale_list
        self.images = {}
        self.images_inv = {}
        for to_sz in scale_list:
            self.images[to_sz] = []
            self.images_inv[to_sz] = []


        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(flags.num_gpus):

                with tf.device('/gpu:%d' % (i)):
                    with tf.name_scope('%s_%d' % ('gpumodel', i)):
                        # create image placeholder for down_sampling
                        from_sz = scale_list[-1]
                        self.images[from_sz].append(tf.placeholder(tf.float32, shape=[None, from_sz, from_sz, 3]))
                        for to_sz in scale_list[0:-1]:
                            self.images[to_sz].append(self.build_Q(self.images[from_sz][i], from_sz, to_sz)) # build image placeholder for up_sampling
                        from_sz = scale_list[0]
                        self.images_inv[from_sz].append(tf.placeholder(tf.float32, shape = [None, from_sz, from_sz, 3]))

                        for to_sz in scale_list[1:]:
                            self.images_inv[to_sz].append(self.build_Q_inv(self.images_inv[from_sz][i], from_sz, to_sz))
                            from_sz = to_sz




        if (flags.prefetch):
            files = glob(os.path.join('./data', flags.dataset_name, flags.input_pattern))
            self.data = par_imread(files, flags.image_size, flags.num_threads)
        else:
            self.data = glob(os.path.join('./data', flags.dataset_name, flags.input_pattern))
        self.build_model(flags)


    def build_Q(self, input, from_size, to_size, reuse = False):
        assert from_size % to_size == 0, "Multigrid model setup error: the from_size({}) should be divisible by to_size({})".format(from_size, to_size)
        var_scope = 'transfer_{}_{}'.format(from_size, to_size)
        filter_size = from_size / to_size
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                scope.reuse_variables()

            ratio_value = np.float32(1.0 / filter_size / filter_size)
            temp_w = np.zeros((filter_size, filter_size, 3, 3), np.float32)
            temp_w[:,:,0,0] = ratio_value
            temp_w[:,:,1,1] = ratio_value
            temp_w[:,:,2,2] = ratio_value
            Q = tf.Variable(temp_w, name=var_scope + '_Q', trainable = False)

            # data format：[batch, height, width, channels]
            down_sampled = tf.nn.conv2d(input, Q, [1, filter_size, filter_size, 1], padding='SAME')

            return down_sampled

    def build_Q_inv(self, input, from_size, to_size, reuse = False):
        assert to_size % from_size == 0, "Multigrid model setup error: the to_size({}) should be divisible by from_size({})".format(to_size, from_size)
        var_scope = 'transfer_inv_{}_{}'.format(from_size, to_size)
        filter_size =  to_size / from_size
        with tf.variable_scope(var_scope) as scope:
            if reuse:
                scope.reuse_variables()

            temp_w_inv = np.zeros((filter_size, filter_size, 3, 3), np.float32)
            temp_w_inv[:,:,0,0] = 1
            temp_w_inv[:,:,1,1] = 1
            temp_w_inv[:,:,2,2] = 1
            Q_inv = tf.Variable(temp_w_inv, name=var_scope + '_Qinv', trainable = False)
            batch_size = tf.shape(input)[0]
            deconv_shape = [batch_size, to_size, to_size, 3]
            up_sampled = tf.nn.conv2d_transpose(input, Q_inv, output_shape=deconv_shape, strides=[1,filter_size,filter_size,1])
            return up_sampled
    # This function was get from the tensorflow tutorial of Cifar10 multi-gpu version
	# It is used to calculate the average of gradient get from each tower

    def average_gradient(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def clip_by_abs(self, t, value):
        maxvalue = tf.reduce_max(tf.abs(t))
        clip_t = tf.cond(maxvalue > value, lambda: t/maxvalue * value, lambda: t)
        return clip_t

    def shrink_gard(self, grad_var):
        update_grad_var = []
        for grad, var in grad_var:
            if 'fc' in var.name:
                update_grad_var.append((0.1*grad, var))
            elif 'des64' in var.name:
                if 'h0' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[64][0] * grad, var))
                elif 'h1' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[64][1] * grad, var))
                elif 'h2' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[64][2] * grad, var))
                else:
                    update_grad_var.append((grad, var))

            elif 'des16' in var.name:
                if 'h0' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[16][0] * grad, var))
                elif 'h1' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[16][1] * grad, var))
                elif 'h2' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[16][2] * grad, var))
                elif 'h3' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[16][3] * grad, var))
                else:
                    update_grad_var.append((grad, var))

            elif 'des4' in var.name:
                if 'h0' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[4][0] * grad, var))
                elif 'h1' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[4][1] * grad, var))
                elif 'h2' and 'kernel' in var.name:
                    update_grad_var.append((1.0 / self.fsz[4][2] * grad, var))
                else:
                    update_grad_var.append((grad, var))
            else:
                update_grad_var.append((grad, var))

        return update_grad_var


    def build_model(self, flags):
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.original_images = {}
        self.synthesized_images = {}
        self.m_original = {}
        self.m_synthesized = {}
        self.sample_loss = {}
        self.vars = {}
        self.train_loss = {}
        self.recon_loss = {}
        self.grad_op = {}
        self.langevin_update = {}
        m_optim = {}
        grads_and_vars = {}
        clipped_grads_and_vars = {}
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.decay_steps = flags.decay_steps * (len(self.scale_list) - 1) * int(math.ceil(float(len(self.data)) /  self.batch_size))
        self.lr = tf.train.exponential_decay(flags.learning_rate, self.global_step, self.decay_steps, flags.decay_rate, staircase=True)

        self.fsz = {}
        self.netdepth = {}
		
        for im_sz in self.scale_list[1:]:
            self.original_images[im_sz] = []
            self.synthesized_images[im_sz] = []
            self.m_original[im_sz] = []
            self.m_synthesized[im_sz] = []
            self.sample_loss[im_sz] = []
            self.vars[im_sz] = []
            self.train_loss[im_sz] = []
            self.recon_loss[im_sz] = []
            self.grad_op[im_sz] = []
            self.langevin_update[im_sz] = []
            grads_and_vars[im_sz] = []
            clipped_grads_and_vars[im_sz] = []
            #m_optim[im_sz] = tf.train.AdamOptimizer(flags.learning_rate, beta1 = flags.beta1) #define optimizer and training option
            m_optim[im_sz] = tf.train.MomentumOptimizer(self.lr, momentum = flags.beta1)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(flags.num_gpus):
                varreuse = (i != 0) #use this to make sure that models running on different gpu share the same parameters
                with tf.device('/gpu:%d' % (i)):
                    with tf.name_scope('%s_%d' % ('gpumodel', i)) as scope:
                        for im_sz in self.scale_list[1:]:
                            self.original_images[im_sz].append(tf.placeholder(tf.float32, shape = [None, im_sz, im_sz, 3]))
                            self.synthesized_images[im_sz].append(tf.placeholder(tf.float32, shape = [None, im_sz, im_sz, 3]))
                            self.m_original[im_sz].append(self.descriptor_warpper(self.original_images[im_sz][i], self.phase, im_sz, varreuse))
                            self.m_synthesized[im_sz].append(self.descriptor_warpper(self.synthesized_images[im_sz][i], self.phase, im_sz, True))

            t_vars = tf.trainable_variables()


            for i in range(flags.num_gpus):
                with tf.device('/gpu:%d' % (i)):
                    with tf.name_scope('%s_%d' % ('gpumodel', i)) as scope:
                        for im_sz in self.scale_list[1:]:
                            self.vars[im_sz].append([var for var in t_vars if 'des{}_'.format(im_sz) in var.name])


                            self.sample_loss[im_sz].append(tf.reduce_sum(self.m_synthesized[im_sz][i]))
                            ## To maximize the log-likelihood, w-8.13556e-05e minimize the negative log-likelihood:
                            # ## grad = grad( tf.reduce_sum((self.m_64_synthesized) - tf.reduce_sum((self.m_64_original) )
                            self.train_loss[im_sz].append(tf.subtract(tf.reduce_mean(self.m_synthesized[im_sz][i]), tf.reduce_mean(self.m_original[im_sz][i])))

                            self.recon_loss[im_sz].append(tf.reduce_mean(
                            tf.abs(tf.subtract(self.original_images[im_sz][i], self.synthesized_images[im_sz][i]))))
                            # define gradient update and clipping policy
                            grads_and_vars[im_sz].append(m_optim[im_sz].compute_gradients(self.train_loss[im_sz][i], var_list = self.vars[im_sz][i]))

                            grads_and_vars[im_sz][i] = self.shrink_gard(grads_and_vars[im_sz][i])

                            #grads_and_vars[im_sz][i] = [(0.1 * grad, var) if 'fc' in var.name else (grad, var) for grad, var in grads_and_vars[im_sz][i]]
                            clipped_grads_and_vars[im_sz].append([(self.clip_by_abs(grad, flags.clip_grad), var)
                                                                      if ('kernel' in var.name or 'batch' in var.name) else (grad, var) for grad, var in grads_and_vars[im_sz][i]])
                            #  define Langevin sampling op
                            self.grad_op[im_sz].append(tf.gradients(self.sample_loss[im_sz][i], self.synthesized_images[im_sz][i])[0])
                            self.langevin_update[im_sz].append(get_langevin_update(self.grad_op[im_sz][i], self.synthesized_images[im_sz][i], im_sz, flags))



        for grad, var in grads_and_vars[64][0]:
            tf.summary.scalar(var.name + "_maxgrad", tf.reduce_max(tf.abs(grad)))
            tf.summary.scalar(var.name + "_meangrad", tf.reduce_mean(tf.abs(grad)))
            tf.summary.scalar(var.name + "_gard", tf.norm(grad))

        for grad, var in clipped_grads_and_vars[64][0]:
            tf.summary.scalar(var.name + "_clip_maxgrad", tf.reduce_max(tf.abs(grad)))
            tf.summary.scalar(var.name + "_clip_meangrad", tf.reduce_mean(tf.abs(grad)))
            tf.summary.scalar(var.name + "_clip_gard", tf.norm(grad))


        self.saver = tf.train.Saver()
        self.train_op = {}
        total_grads = {}
        for im_sz in self.scale_list[1:]:
            total_grads[im_sz] = self.average_gradient(clipped_grads_and_vars[im_sz])
            self.train_op[im_sz] = m_optim[im_sz].apply_gradients(total_grads[im_sz], global_step = self.global_step)
            #self.train_op[im_sz] = m_optim[im_sz].apply_gradients(clipped_grads_and_vars[im_sz][0])






    def descriptor_warpper(self, inputs, phase, im_sz, reuse = False):
        if im_sz == 64:
            return self.descriptor64(inputs, phase, reuse)
        elif im_sz == 16:
            return self.descriptor16(inputs, phase, reuse)
        elif im_sz == 4:
            return self.descriptor4(inputs, phase, reuse)
        else:
            print('Error!! unsuportted model version {}'.format(im_sz))
            exit()


    def descriptor64(self, inputs, phase, reuse = False):
        net_name = 'des64'
        with tf.variable_scope(net_name) as scope:
            if reuse:
                scope.reuse_variables()
            kernel_reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            kernel_init = tf.contrib.layers.xavier_initializer(True)
            ## layer 1 5x5, stride = 2, pad = 2, n_out = 96
            h0 = conv_layer(inputs, 64, 5, 2, phase, kernel_reg, kernel_init, net_name, 0)
            ## layer 2 3x3, stride = 2, pad = 2, n_out = 256
            h1 = conv_layer(h0, 128, 3, 2, phase, kernel_reg, kernel_init, net_name, 1)
            ## layer 3 3x3, stride = 1, pad = 2, n_out = 256
            h2 = conv_layer(h1, 256, 3, 1, phase, kernel_reg, kernel_init, net_name, 2)


            if reuse == False:
                self.fsz[64] = [h0.shape[2].value * h0.shape[1].value, h1.shape[2].value * h1.shape[1].value, h2.shape[2].value * h2.shape[1].value]
                self.netdepth[64] = 3
                '''
                print('64')
                print(self.fsz[64])
                '''


            ## layer 4 fully connected, out = 1
            num_out = int(h2.shape[1] * h2.shape[2] * h2.shape[3])
            h3 = tf.layers.dense(tf.reshape(h2, [-1, num_out], name=net_name+'_reshape'), 1,
                                 kernel_regularizer=kernel_reg, kernel_initializer = kernel_init, name=net_name+'_h3_fc')
            return h3

    def descriptor16(self, inputs, phase, reuse = False):
        net_name = 'des16'
        with tf.variable_scope(net_name) as scope:
            if reuse:
                scope.reuse_variables()

            kernel_reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            kernel_init = tf.contrib.layers.xavier_initializer(True)
            ## layer 1 5x5, stride = 2, pad = 2, n_out = 96
            h0 = conv_layer(inputs, 96, 5, 2, phase, kernel_reg, kernel_init, net_name, 0)
            ## layer 2 3x3, stride = 1, pad = 2, n_out = 256
            h1 = conv_layer(h0, 128, 3, 1, phase, kernel_reg, kernel_init, net_name, 1)
            ## layer 3 3x3, stride = 1, pad = 2, n_out = 256
            h2 = conv_layer(h1, 256, 3, 1, phase, kernel_reg, kernel_init, net_name, 2)
            ## layer 4 3x3, stride = 1, pad = 2, n_out = 256
            h3 = conv_layer(h2, 512, 3, 1, phase, kernel_reg, kernel_init, net_name, 3)


            if reuse == False:
                self.fsz[16] = [h0.shape[2].value * h0.shape[1].value, h1.shape[2].value * h1.shape[1].value, h2.shape[2].value * h2.shape[1].value, h3.shape[2].value * h3.shape[1].value]
                self.netdepth[16] = 4
                '''
                print('16')
                print(self.fsz[16])
                '''


            ## layer 5 fully connected, out = 1
            num_out = int(h3.shape[1] * h3.shape[2] * h3.shape[3])
            h4 = tf.layers.dense(tf.reshape(h3, [-1, num_out], name=net_name+'_reshape'), 1,
                                 kernel_regularizer=kernel_reg, kernel_initializer = kernel_init, name=net_name+'_h4_fc')

            return h4

    def descriptor4(self, inputs, phase, reuse = False):
        net_name = 'des4'
        with tf.variable_scope(net_name) as scope:
            if reuse:
                scope.reuse_variables()

            kernel_reg = tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
            kernel_init = tf.contrib.layers.xavier_initializer(True)
            ## layer 1 5x5, stride = 2, pad = 2, n_out = 96
            h0 = conv_layer(inputs, 96, 5, 2, phase, kernel_reg, kernel_init, net_name, 0)
            ## layer 2 3x3, stride = 1, pad = 2, n_out = 256
            h1 = conv_layer(h0, 128, 3, 1, phase, kernel_reg, kernel_init, net_name, 1)
            ## layer 3 3x3, stride = 1, pad = 2, n_out = 256
            h2 = conv_layer(h1, 256, 3, 1, phase, kernel_reg, kernel_init, net_name, 2)

            if reuse == False:
                self.fsz[4] = [h0.shape[2].value * h0.shape[1].value, h1.shape[2].value * h1.shape[1].value, h2.shape[2].value * h2.shape[1].value]
                self.netdepth[4] = 3
                '''
                print('4')
                print(self.fsz[4])
                '''

            ## layer 4 fully connected, out = 1
            num_out = int(h2.shape[1] * h2.shape[2] * h2.shape[3])
            h3 = tf.layers.dense(tf.reshape(h2, [-1, num_out], name=net_name+'_reshape'), 1,
                                 kernel_regularizer=kernel_reg, kernel_initializer = kernel_init, name=net_name+'_h3_fc')

            return h3


    def Langevin_sampling(self, langevin_update, samples, to_sz, flags):
        for t in xrange(flags.T):

            tmp_feed_dict = {}
            tmp_feed_dict[self.phase] = True
            for small_idx in xrange(flags.num_gpus):
                tmp_feed_dict[self.synthesized_images[to_sz][small_idx]] = samples[small_idx]


            #grad = self.sess.run(grad_op, feed_dict = tmp_feed_dict)
            samples = self.sess.run(langevin_update, feed_dict=tmp_feed_dict)


        return samples

    def train(self, flags):
        self.sess.run(tf.global_variables_initializer())

        self.mysummary = tf.summary.merge_all()

        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
        self.sess.graph.finalize()
        counter = 1
        start_time = time.time()

        batch_idxs = int(math.ceil(float(len(self.data)) / self.batch_size))
        small_batchs = int(math.ceil(float(self.batch_size) / flags.num_gpus))  # calculate the number of images processed on each gpu
        burst_len = (self.batch_size * flags.read_len)

        could_load, checkpoint_counter = self.load(flags)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
            startepoch = int(math.floor(float(counter - 1) / batch_idxs))
            startidx_batch = np.mod((counter - 1), batch_idxs)
            self.global_step = (counter - 1) * (len(self.scale_list) - 1)

            if (flags.prefetch == False):
                start_idx = int(math.floor(float(startidx_batch) / burst_len)) * burst_len
                end_idx = min(start_idx + burst_len, len(self.data))
                files = self.data[start_idx: end_idx]
                tmp_list = par_imread(files, flags.image_size, flags.num_threads)

        else:
            print(" [!] Load failed...")
            startidx_batch = 0
            startepoch = 0






        for epoch in xrange(startepoch, flags.epoch):


            for idx_batch in xrange(startidx_batch, batch_idxs):
                samples = {}
                batch_images = []

                print ((counter, self.sess.run(self.lr)))

                if (flags.prefetch == False and np.mod(idx_batch, flags.read_len) == 0):
                    start_idx = idx_batch * self.batch_size
                    end_idx = min(start_idx + burst_len, len(self.data))
                    files = self.data[start_idx: end_idx]
                    tmp_list = par_imread(files, flags.image_size, flags.num_threads)

                if (idx_batch == batch_idxs - 1):
                    cursmall_batches =  int(math.ceil(float(len(self.data) - idx_batch * self.batch_size)/flags.num_gpus))
                else:
                    cursmall_batches = small_batchs


                for small_idx in xrange(flags.num_gpus):
                    start_idx = idx_batch * self.batch_size + small_idx * cursmall_batches
                    end_idx = min(idx_batch * self.batch_size + (small_idx + 1) * cursmall_batches, len(self.data))
                    if flags.prefetch:
                        batch_images.append(np.array(self.data[start_idx : end_idx]).astype(np.float32))
                    else:
                        start_idx = np.mod(start_idx, burst_len)
                        end_idx = np.mod(end_idx,burst_len)
                        if (end_idx == 0):
                            end_idx = burst_len
                        batch_images.append(np.array(tmp_list[start_idx: end_idx]).astype(np.float32))

                        #files = self.data[start_idx : end_idx]
                        #list_images = par_imread(files, flags.image_size, flags.num_threads)
                        #batch_images.append(np.array(list_images).astype(np.float32))


                train_images = {}

                #print(time.time() - start_time)
                # generate initial samples
                to_sz = self.scale_list[0]
                from_sz = self.scale_list[-1]

                tmp_feed_dict = {}
                for small_idx in xrange(flags.num_gpus):
                        tmp_feed_dict[self.images[from_sz][small_idx]] = batch_images[small_idx]


                #samples[to_sz] = np.array(self.sess.run(self.images[to_sz], feed_dict = tmp_feed_dict)).astype(np.float32)
                tmp =  self.sess.run(self.images[to_sz], feed_dict = tmp_feed_dict)
                samples[to_sz] = []
                for small_idx in xrange(flags.num_gpus):
                    samples[to_sz].append(np.array(tmp[small_idx]).astype(np.float32))

                from_sz = to_sz

                for to_sz in self.scale_list[1:]:
                        # training images in this scale
                    if to_sz != self.scale_list[-1]:
                        tmp_feed_dict = {}
                        for small_idx in xrange(flags.num_gpus):
                            tmp_feed_dict[self.images[self.scale_list[-1]][small_idx]] = batch_images[small_idx]
                        tmp = self.sess.run(self.images[to_sz], feed_dict = tmp_feed_dict)
                        train_images[to_sz] = []
                        for small_idx in xrange(flags.num_gpus):
                            train_images[to_sz].append(np.array(tmp[small_idx]).astype(np.float32))

                    else:
                        train_images[to_sz] = batch_images

                    #save_images(train_images[to_sz][0],'./{}/input_{}_{:02d}_{:06d}_gpu0.jpg'.format(flags.sample_dir, to_sz, epoch, idx_batch))
                    #save_images(train_images[to_sz][1],'./{}/input_{}_{:02d}_{:06d}_gpu1.jpg'.format(flags.sample_dir, to_sz, epoch, idx_batch))


                    # up sampling
                    tmp_feed_dict = {}
                    for small_idx in xrange(flags.num_gpus):
                        tmp_feed_dict[self.images_inv[from_sz][small_idx]] = samples[from_sz][small_idx]
                    self.sess.run(self.images_inv[to_sz], feed_dict= tmp_feed_dict)

                    tmp = self.sess.run(self.images_inv[to_sz], feed_dict= tmp_feed_dict)
                    tmpsamples = []
                    for small_idx in xrange(flags.num_gpus):
                        tmpsamples.append(np.array(tmp[small_idx]).astype(np.float32))


                    ## run Langevin sampling to recover image
                    #samples[to_sz] = self.Langevin_sampling(self.grad_op[to_sz], tmpsamples, to_sz, flags)

                    samples[to_sz] = self.Langevin_sampling(self.langevin_update[to_sz], tmpsamples, to_sz, flags)


                    # merge samples from all gpus together
                    all_samples = samples[to_sz][0]
                    for i in xrange(1, flags.num_gpus):
                        all_samples = np.append(all_samples, samples[to_sz][i], axis = 0)


                    ## Compute reconstruction error
                    tmp_feed_dict = {}
                    tmp_feed_dict[self.phase] = True
                    for small_idx in xrange(flags.num_gpus):
                        tmp_feed_dict[self.original_images[to_sz][small_idx]] = train_images[to_sz][small_idx]
                        tmp_feed_dict[self.synthesized_images[to_sz][small_idx]] = samples[to_sz][small_idx]

                    err_list = self.sess.run(self.train_loss[to_sz], feed_dict= tmp_feed_dict)
                    err = np.mean(err_list)
                    err_list2 = self.sess.run(self.recon_loss[to_sz], feed_dict=tmp_feed_dict)
                    err2 = np.mean(err_list2)


                    if (epoch < 10):
                        if (np.mod(counter, 755) == 1):
                            #save_images(batch_images[0],'./{}/input_{}_{:02d}_{:06d}.jpg'.format(flags.sample_dir, to_sz, epoch, idx_batch))
                            save_images(all_samples, './{}/train_multigrid_{}_{:02d}_{:06d}.jpg'.format(flags.sample_dir, to_sz, epoch, idx_batch))  ##leave this here for final
                    elif (epoch < 20):
                        if (np.mod(counter, 755) == 1):
                            save_images(all_samples, './{}/train_multigrid_{}_{:02d}_{:06d}.jpg'.format(flags.sample_dir, to_sz, epoch, idx_batch))
                    elif (epoch < 30):
                        if (np.mod(counter, 755) == 1):
                            save_images(all_samples,'./{}/train_multigrid_{}_{:02d}_{:06d}.jpg'.format(flags.sample_dir, to_sz, epoch,  idx_batch))
                    elif (np.mod(counter, 755) == 1):
                            save_images(all_samples, './{}/train_multigrid_{}_{:02d}_{:06d}.jpg'.format(flags.sample_dir, to_sz, epoch, idx_batch))

                    print('Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f} , train_loss[multigrid_{}]: {:.5f}, reconstruction loss[multigrid_{}]: {:.5f}'.format(epoch + 1, idx_batch + 1, batch_idxs, time.time() - start_time, to_sz, err, to_sz, err2))
                    from_sz = to_sz

                tmp_feed_dict = {}
                tmp_feed_dict[self.phase] = True
                ## update multigrid
                for to_sz in self.scale_list[1:]:
                    for small_idx in xrange(flags.num_gpus):
                        tmp_feed_dict[self.original_images[to_sz][small_idx]] = train_images[to_sz][small_idx]
                        tmp_feed_dict[self.synthesized_images[to_sz][small_idx]] = samples[to_sz][small_idx]

                    if (to_sz == 64):
                        summary = self.sess.run(self.mysummary, feed_dict=tmp_feed_dict)
                        self.writer.add_summary(summary, counter)

                    self.sess.run(self.train_op[to_sz], feed_dict = tmp_feed_dict)





                #self.writer.add_summary(summary_str, counter)
                counter += 1

                if (epoch < 10):
                    if np.mod(counter, 755) == 2 or epoch == flags.epoch-1 and idx_batch == batch_idxs-1:
                        self.save(flags, counter)
                elif (epoch < 20):
                    if np.mod(counter, 755) == 2 or epoch == flags.epoch-1 and idx_batch == batch_idxs-1:
                        self.save(flags, counter)
                elif (epoch < 30):
                    if np.mod(counter, 755) == 2 or epoch == flags.epoch-1 and idx_batch == batch_idxs-1:
                        self.save(flags, counter)

                elif np.mod(counter, 755) == 2 or epoch == flags.epoch-1 and idx_batch == batch_idxs-1:
                        self.save(flags, counter)
            startidx_batch = 0


    def model_dir(self, flags):
        return '{}_{}'.format(flags.dataset_name, self.batch_size)

    def save(self, flags, step):
        model_name = 'multigrid.model'
        checkpoint_dir = os.path.join(flags.checkpoint_dir, self.model_dir(flags))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)

    def load(self, flags):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(flags.checkpoint_dir, self.model_dir(flags))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



