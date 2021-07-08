# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from tf_util import mlp, mlp_conv, earth_mover, add_train_summary, add_valid_summary
from pc_distance import tf_approxmatch
import os
import sys
sys.path.append('/media/hp2/EA2ADCA32ADC6E57/Pointcloud/pcn/sampling/')
sys.path.append('/media/hp2/EA2ADCA32ADC6E57/Pointcloud/pcn/grouping/')

from tf_grouping import knn_point,group_point
from tf_sampling import gather_point
class Model:
    def __init__(self, inputs, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4

        
        self.features = self.create_encoder(inputs)
        self.coarse,self.com_points,self.out_label,self.o_points = self.create_decoder(self.features,inputs)
            
        self.loss, self.update = self.create_loss(self.coarse, self.com_points, self.out_label, gt, alpha)
        self.outputs = self.out_label
        self.visualize_ops = [inputs[0], self.coarse[0],self.o_points[0], gt[0]]
        self.visualize_titles = ['input', 'coarse', 'sample', 'gt']

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
            
            coarse_label=tf.ones(shape=[tf.shape(inputs)[0],self.num_coarse,1])*(-1)
            input_label=tf.ones(shape=[tf.shape(inputs)[0],tf.shape(inputs)[1],1])
            
            com_points= tf.concat([coarse,inputs],1) #batch_size * ndataset * 3
            label2=tf.concat([coarse_label,input_label],1)
       
            inpoints2=tf.concat([com_points,label2],2)
            
        
            
            
        with tf.variable_scope('fdense', reuse=tf.AUTO_REUSE):     
            distance, indexs = knn_point(16, com_points,com_points)
            density=group_point(com_points,indexs)
            density=tf.reduce_mean(tf.exp(-distance/(0.034**2)),2)
            density=tf.expand_dims(density,2)
            density=group_point(density,indexs) #in 4 512 1-----4 512 16 1
        
            den=mlp_conv(density,[128,256])
            
        with tf.variable_scope('flocal', reuse=tf.AUTO_REUSE):     
            local_patches=group_point(com_points,indexs)
            local_patches=local_patches-tf.expand_dims(com_points,2)
            local=mlp_conv(local_patches,[128,256])
            
        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            
            feat=mlp_conv(inpoints2,[128,256])
            feat=group_point(feat,indexs)
            
            feat=tf.multiply(feat,den)
            feat=tf.multiply(feat,local)
            feat=tf.reduce_sum(feat,2)
        
            
        with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inpoints2, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, self.num_coarse+tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [256])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
            
            features = tf.tile(tf.expand_dims(features,1),[1,self.num_coarse+tf.shape(inputs)[1],1])
        with tf.variable_scope('folding2', reuse=tf.AUTO_REUSE): 
            feat=mlp_conv(feat,[256])
            feat=tf.concat([feat,features],2)
        with tf.variable_scope('folding3', reuse=tf.AUTO_REUSE): 
            feat=mlp_conv(feat,[128,32,2])
            
            lab=tf.slice(feat,[0,0,1],[com_points.shape[0],com_points.shape[1],1])
            lab=tf.squeeze(lab,2)
            values,index=tf.nn.top_k(lab,1024)
            o_points=gather_point(com_points,index)
            
            out_label=tf.reshape(feat,[-1,2])
            
            
        
        
      
        return coarse,com_points,out_label,o_points

        

    def create_loss(self, coarse, com_points,out_label, gt, alpha):
        gt_ds = gt[:, :coarse.shape[1], :]
        loss_coarse = earth_mover(coarse, gt_ds)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)
        
        
        match = tf_approxmatch.approx_match(gt_ds,com_points)
        match=(tf.sign(match-0.6)+1)*0.5
        match=tf.reduce_max(match,2)
        label=tf.one_hot(tf.cast(match,tf.int32),2)
        label=tf.reshape(label,[-1,2])
        loss_sample = tf.losses.softmax_cross_entropy(label, out_label)
        add_train_summary('train/fine_loss', loss_sample)
        update_fine = add_valid_summary('valid/fine_loss', loss_sample)

        loss = loss_coarse + alpha * loss_sample
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]



#tf.nn.top_k
