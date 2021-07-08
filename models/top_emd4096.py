# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from tf_util import mlp, mlp_conv, chamfer, earth_mover, add_train_summary, add_valid_summary


class Model:
    def __init__(self, inputs, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs)
        self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0],  self.fine[0], gt[0]]
        self.visualize_titles = ['input', 'fine output', 'ground truth']

    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
        return features


    def create_level(self, level, input_channels, output_channels, inputs, bn):
        tarch=[4,8,8,16]
        with tf.variable_scope('level_%d' % (level), reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [input_channels, input_channels/2,input_channels/4, input_channels/8,output_channels*tarch[level]], bn=bn)
            features = tf.reshape(features, [tf.shape(features)[0], -1, output_channels])
        return features
        
   
    def create_decoder(self, features):
        
        Nout=1024
      
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            level0 = mlp(features, [256, 64, 1024* 4])
            level0 = tf.tanh(level0, name='tanh_0')
            level0 = tf.reshape(level0, [-1,4, 1024])
            outs = [level0, ]  
            bn=None
            
            for i in range(1, 4):
                if i == 3:
                    Nout = 3
                    bn = None
            
                inp = outs[-1]
                y = tf.expand_dims(features, 1)
                y = tf.tile(y, [1, tf.shape(inp)[1], 1])
                y = tf.concat([inp, y], 2)
                
                
                outs.append(tf.tanh(self.create_level(i, 2048, Nout, y, bn), name='tanh_'+str(i)))
        fine=outs[-1]
        fine=tf.reshape(fine,[tf.shape(fine)[0],4096,3])
        return fine
        
        
        
        
        
    def create_loss(self, fine, gt, alpha):
        
        
        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = alpha*loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_fine, update_loss]
