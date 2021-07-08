# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from tf_util import mlp, mlp_conv, chamfer, earth_mover, add_train_summary, add_valid_summary
import os
import sys
sys.path.append('/media/hp2/EA2ADCA32ADC6E57/Pointcloud/pcn/sampling/')
sys.path.append('/media/hp2/EA2ADCA32ADC6E57/Pointcloud/pcn/grouping/')
from tf_sampling import farthest_point_sample,gather_point
from tf_grouping import knn_point,group_point

class Model:
    def __init__(self, inputs, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs)
        self.coarse, self.coarse2,self.fine = self.create_decoder(self.features,inputs)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.fine[0], gt[0]]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
        return features

    def create_decoder(self, features,inputs):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])
            
            #coarse2= tf.concat([coarse,inputs],1)
            #dada=farthest_point_sample(1024,coarse2)
            coarse2=coarse#gather_point(coarse2,dada)
            
        
            
        '''    
        with tf.variable_scope('fdense', reuse=tf.AUTO_REUSE):     
            distance, indexs = knn_point(16, coarse2,coarse2)
            density=group_point(coarse2,indexs)
            density=tf.reduce_mean(tf.exp(-distance/(0.034**2)),2)
            density=tf.expand_dims(density,2)
            density=group_point(density,indexs) #in 4 512 1-----4 512 16 1
        
            den=mlp_conv(density,[128,256])
           
        with tf.variable_scope('flocal', reuse=tf.AUTO_REUSE):     
            local_patches=group_point(coarse2,indexs)
            local_patches=local_patches-tf.expand_dims(coarse2,2)
            local=mlp_conv(local_patches,[128,256])
        '''    
        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            distance, indexs = knn_point(16, coarse2,coarse2)
            feat=mlp_conv(coarse2,[128,256])
            feat=group_point(feat,indexs)
            
            #feat=tf.multiply(feat,den)
            #feat=tf.multiply(feat,local)
            feat=tf.reduce_sum(feat,2)
        
            
        with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE):
            features = mlp_conv(coarse2, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, 1024, 1])], axis=2)
        with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
            
            features = tf.tile(tf.expand_dims(features,1),[1,1024,1])
        with tf.variable_scope('folding2', reuse=tf.AUTO_REUSE): 
            feat=mlp_conv(feat,[512])
            feat=tf.concat([feat,features],2)
        with tf.variable_scope('folding3', reuse=tf.AUTO_REUSE): 
            feat=mlp_conv(feat,[512,256,6])
        fine=tf.reshape(feat,[tf.shape(inputs)[0],2048,3])
            
            
            
      
        return coarse,coarse2, fine

    def create_loss(self, coarse, fine, gt, alpha):
        gt_ds = gt[:, :coarse.shape[1], :]
        loss_coarse = earth_mover(coarse, gt_ds)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = earth_mover(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss = loss_coarse + alpha * loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]
