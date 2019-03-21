#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import math
import utils
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

from tqdm import tqdm
from dataFunctions import *
from keras import backend as K
from models import single_model
from models import semantic_model
from tensorflow.contrib import slim

class Conf(object):
    def __init__(self, params=None, mode=None):
        """
        Initializes baseline by setting params, mode,
        and loading mean values to be used in the case non-RGB imagery is being used
        :param params: input parameters from params.py
        :param mode: either SEMANTIC_MODE or SINGLEVIEW_MODE from params (0 or 1)
        :return: None
        """
        self.params=params
        self.mode = mode
        self.meanVals = None
        if params.NUM_CHANNELS != 3:
            if params.MEAN_VALS_FILE is None:
                self.meanVals = np.zeros(params.NUM_CHANNELS).tolist()
            else:
                self.meanVals = json.load(open(params.MEAN_VALS_FILE))

        # TensorFlow allocates all GPU memory up front by default, so turn that off
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True


    def image_generator(self, trainData):
        """
        Generates training batches of image data and ground truth from either semantic or depth files
        :param trainData: training paths of CLS files (string replacement to get RGB and depth files)
        and starting x,y pixel positions (can be non-zero if blocking is set to happen in params.py)
        :yield: current batch data
        """
        idx = np.random.permutation(len(trainData))
        while True:
            batchInds = get_batch_inds(idx, self.params)
            for inds in batchInds:
                imgBatch, labelBatch = load_batch(inds, trainData, self.params, self.mode, self.meanVals)
                yield (imgBatch, labelBatch)


    def train(self):
        """
        Launches training and stores checkpoints at a frequency defined within params
        :return: None
        """
        image_paths = get_image_paths(self.params, isTest=False)
        if self.params.BLOCK_IMAGES:
            train_data = get_train_data(image_paths, self.params)
        else:
            train_data = []
            for imgPath in image_paths:
                train_data.append((imgPath, 0, 0))

        # training batch generator
        train_datagen = self.image_generator(train_data)

        # build training model
        self.build_model(training = True)

        # define ckpt saver
        if self.params.CONTINUE_TRAINING:
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = self.params.MODEL_MAX_KEEP)

        # define summary writer
        summary_writer = tf.summary.FileWriter(self.params.CHECKPOINT_DIR, tf.get_default_graph())

        with tf.Session(config=self.config) as sess:
            # do initialization
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # load pre-trained backbone model
            if self.params.PRETRAINED_MODEL is not None:
                # select exclude items according to input types
                if self.params.NUM_CHANNELS == 3:
                    exclude_list = [self.params.BACKBONE + "/logits"]
                elif self.params.NUM_CHANNELS == 8:
                    exclude_list = [self.params.BACKBONE + "/logits", self.params.BACKBONE + "/conv1"]
                else:
                    exclude_list = []

                variables_to_restore = slim.get_variables_to_restore(exclude = exclude_list)
                variable_restore_op = slim.assign_from_checkpoint_fn(self.params.PRETRAINED_MODEL,
                                                                     variables_to_restore,
                                                                     ignore_missing_vars=True)
                variable_restore_op(sess)
            else:
                pass

            # load continue model if it is set
            if self.params.CONTINUE_TRAINING:
                if self.mode == self.params.SINGLEVIEW_MODE:
                    saver.restore(sess, self.params.CONTINUE_SINGLEVIEW_MODEL_FILE)
                elif self.mode == self.params.SEMANTIC_MODE:
                    saver.restore(sess, self.params.CONTINUE_SEMANTIC_MODEL_FILE)
                else:
                    pass
            else:
                pass

            # calculate the number of batches per epoch
            steps_per_epoch = int(len(train_data)/self.params.BATCH_SZ)

            start_time = time.time()
            for epoch in range(self.params.NUM_EPOCHS):
                for idx in range(steps_per_epoch):
                    # get feed batch
                    img_batch, gt_batch = next(train_datagen)

                    result = sess.run(self.fetches, feed_dict={self.input_tensor: img_batch,
                                                               self.gt_tensor: gt_batch})

                    # get current global step
                    gs = result["global_step"]

                    # make sure the loss will not be 'NAN'
                    assert (not np.isnan(result["loss"])), "Model diverged with loss = NAN"

                    # save summary every SUMMARY_SAVE_FREQ iterations
                    if gs % self.params.SUMMARY_SAVE_FREQ == 0:
                        summary_writer.add_summary(result["summary"], gs)
                        print("Epoch: [%2d] [%5d/%5d] time: %4.4f loss: %.8f" \
                              % (epoch+1, idx+1, steps_per_epoch, time.time() - start_time, result["loss"]))

                # save ckpt every MODEL_SAVE_PERIOD epoches
                if (epoch+1) % self.params.MODEL_SAVE_PERIOD == 0:
                    ckpt_path = self.params.CHECKPOINT_PATH.format(epoch=epoch+1)
                    print("[*] Saving checkpoint --> " + ckpt_path)
                    saver.save(sess, ckpt_path)


    def test(self):
        """
        Launches testing, which saves output files
        :return: None
        """
        imgPaths = get_image_paths(self.params, isTest=True)
        print("Number of files = {}".format(len(imgPaths)))

        # get corresponding STR according to mode
        if self.mode == self.params.SINGLEVIEW_MODE:
            numPredChannels = 1
            outReplaceStr = self.params.AGLPRED_FILE_STR
        elif self.mode == self.params.SEMANTIC_MODE:
            numPredChannels = self.params.NUM_CATEGORIES
            outReplaceStr = self.params.CLSPRED_FILE_STR
        else:
            pass

        # build test model
        self.build_model(training = False)

        with tf.get_default_graph().as_default():
            # define saver to restore
            saver = tf.train.Saver()

            with tf.Session(config=self.config) as sess:
                # do initialization
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                # restore model
                try:
                    if self.mode == self.params.SINGLEVIEW_MODE:
                        saver.restore(sess, self.params.SINGLEVIEW_TEST_MODEL)
                    elif self.mode == self.params.SEMANTIC_MODE:
                        saver.restore(sess, self.params.SEMANTIC_TEST_MODEL)
                    else:
                        raise RuntimeError('Unkown mode!! (should be SINGLEVIEW or SEMANTIC)')
                except Exception as e:
                    print("Cound not restore checkpoint file!")
                    print("Reason:", e)

                for imgPath in tqdm(imgPaths):
                    imageName = os.path.split(imgPath)[-1]
                    outName = imageName.replace(self.params.IMG_FILE_STR, outReplaceStr)
                    color_outName = outName.replace(self.params.LABEL_FILE_EXT, self.params.COLOR_PRED_EXT)

                    currImg = load_img(imgPath)
                    # perform a contrast stretch if it is not RGB image
                    #if self.params.NUM_CHANNELS != 3:
                    #    currImg = contrast_stretch(currImg)

                    # the batch size of test phase is just 1
                    img = np.expand_dims(currImg, axis=0).astype('float32')
                    img = image_batch_preprocess(img, self.params, self.meanVals)

                    # get prediction
                    result = sess.run(self.fetches, feed_dict={self.input_tensor: img})
                    pred_np = result["pred"][0,:,:,:]
                    color_pred_np = result["color_pred"]

                    if self.mode == self.params.SINGLEVIEW_MODE:
                        pred = pred_np[:,:,0]
                    elif self.mode == self.params.SEMANTIC_MODE:
                        if self.params.NUM_CATEGORIES > 1:
                            pred = np.argmax(pred_np, axis=2).astype('uint8')
                        else:
                            pred = (pred_np > self.params.BINARY_CONF_TH).astype('uint8')

                        if self.params.CONVERT_LABELS:
                            pred = convert_labels(pred, self.params, toLasStandard=True)
                        else:
                            pass
                    else:
                        pass

                    # save results
                    tifffile.imsave(os.path.join(self.params.OUTPUT_DIR, outName), pred, compress=6)
                    mpimg.imsave(os.path.join(self.params.COLOR_OUTPUT_DIR, color_outName), color_pred_np)


    def get_model(self, input_tensor, training = True):
        if self.mode == self.params.SINGLEVIEW_MODE:
            pred_out = single_model(input_tensor, is_training = training)
        elif self.mode == self.params.SEMANTIC_MODE:
            pred_out = semantic_model(input_tensor,
                                      num_classes = self.params.NUM_CATEGORIES,
                                      is_training = training)
        else:
            raise RuntimeError('Unkown mode!! (should be SINGLEVIEW or SEMANTIC)')
        return pred_out


    def build_model(self, training = True):
        """
        Prepares either the single-view depth prediction model or
        the semantic segmentation model depending on params.py
        :return: None
        """
        imgSz = None
        if self.params.BLOCK_IMAGES:
            imgSz = self.params.BLOCK_SZ
        else:
            imgSz = self.params.IMG_SZ

        # define input placeholder
        self.input_tensor = tf.placeholder(tf.float32,
                                           shape = [None, imgSz[0], imgSz[1], self.params.NUM_CHANNELS],
                                           name="inputs")

        # get the number of gt channels, setting loss function
        numChannels = None
        if self.mode == self.params.SINGLEVIEW_MODE:
            loss = self.single_loss
            numChannels = 1
        elif self.mode == self.params.SEMANTIC_MODE:
            loss = self.semantic_loss
            numChannels = self.params.NUM_CATEGORIES+1
        else:
            pass

        # define colorized method
        def colorize_method(value):
            colorized_image = None
            if self.mode == self.params.SINGLEVIEW_MODE:
                colorized_image = utils.single_colorize(value, mask_off=self.params.IGNORE_VALUE, cmap = self.params.SINGLEVIEW_CMAP)
            elif self.mode == self.params.SEMANTIC_MODE:
                colorized_image = utils.semantic_colorize(value)
            else:
                pass
            if self.params.BATCH_SZ == 1 and training == True:
                colorized_image = tf.expand_dims(colorized_image, 0)
            return colorized_image

        # define ground truth placeholder for training
        self.gt_tensor = tf.placeholder(tf.float32,
                                        shape = [None, imgSz[0], imgSz[1], numChannels],
                                        name="gt")

        # get prediction tensor for both training and test
        pred_out = self.get_model(self.input_tensor, training = training)

        if training:
            # define loss tensor
            total_loss = loss(pred_out, self.gt_tensor)

            # define method to calculate colorized diff map
            def diff_with_colorize(value, value_true):
                colorized_diff = None
                if self.mode == self.params.SINGLEVIEW_MODE:
                    colorized_diff = utils.single_colorize(tf.abs(tf.subtract(value, value_true)), mask_off=self.params.IGNORE_VALUE, cmap="jet")
                    if self.params.BATCH_SZ == 1:
                        colorized_diff = tf.expand_dims(colorized_diff, 0)
                elif self.mode == self.params.SEMANTIC_MODE:
                    diff = tf.cast(tf.equal(tf.argmax(value,axis=-1), tf.argmax(value_true,axis=-1)), tf.int32)
                    diff_colormap = np.array([[96, 0, 128], [0, 0, 0]])
                    diff_colormap_tensor = tf.constant(diff_colormap, dtype=tf.uint8)
                    colorized_diff = tf.gather(diff_colormap_tensor, diff)
                else:
                    pass
                return colorized_diff

            # define training variables
            with tf.name_scope("training_variables"):
                # define global step variable
                global_step = tf.Variable(0, name='global_step', trainable = False)

                # define learning rate variable
                lr = tf.Variable(self.params.LEARNING_RATE, trainable = False)

                # define optimizer
                if self.params.OPTIMIZER == "Adam":
                    optim = tf.train.AdamOptimizer(lr)
                else:
                    pass

                # define training operator, for batch norm usage
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies([tf.group(*update_ops)]):
                    train_operation = optim.minimize(total_loss, global_step = global_step)

                # get colorized prediction
                color_pred = colorize_method(pred_out)

                # get colorized ground truth
                color_gt = colorize_method(self.gt_tensor)

                # get colorized diff map
                color_diff = diff_with_colorize(pred_out, self.gt_tensor)

            # define summary operations
            tf.summary.scalar("total_loss", total_loss)
            tf.summary.scalar("learning_rate", lr)
            tf.summary.image("prediction", color_pred)
            tf.summary.image("ground_truth", color_gt)
            tf.summary.image("diff", color_diff)
            # combine summaries
            summary_op = tf.summary.merge_all()

            # declare class inner variables and define training fetches
            self.fetches = {"train": train_operation,
                            "global_step": global_step,
                            "loss": total_loss,
                            "summary": summary_op}
        else:
            with tf.name_scope("test_variables"):
                # get colorized prediction
                color_pred = colorize_method(pred_out)

            # declare class inner variables and define test fetches
            self.fetches = {"pred": pred_out,
                            "color_pred": color_pred}


    def single_loss(self, pred, gt):
        # get valid mask
        mask_true = K.cast(K.not_equal(gt, self.params.IGNORE_VALUE), K.floatx())
        total_loss = None

        def mse_loss():
            masked_squared_error = tf.square(mask_true * (gt - pred))
            masked_mse = tf.reduce_mean(masked_squared_error)
            return masked_mse

        def mae_loss():
            masked_abs_error = tf.abs(mask_true * (gt - pred))
            masked_mae = tf.reduce_mean(masked_abs_error)
            return masked_mae

        def scaleInv_loss():
            map_size = [pred.shape[1].value, pred.shape[2].value]
            map_pixel = map_size[0] * map_size[1]

            # masked different map
            mask_diff = mask_true * (gt - pred)

            # flat maps
            diff_flat = tf.reshape(mask_diff, [-1, map_pixel])

            diff_sqr = tf.square(diff_flat)
            sum_diff_sqr = tf.reduce_sum(diff_sqr, 1)
            sum_diff = tf.reduce_sum(diff_flat, 1)
            sqr_sum_diff = tf.square(sum_diff)

            return tf.reduce_mean(sum_diff_sqr / float(map_pixel) - \
                                  0.5 * sqr_sum_diff / (math.pow(map_pixel, 2)))

        def gradient_loss():
            masked_gt = mask_true * gt
            masked_pred = mask_true * pred

            # compute gradient on x and y axis
            gt_grads_x, gt_grads_y = utils.gradient(masked_gt)
            pred_grads_x, pred_grads_y = utils.gradient(masked_pred)

            diff_x = gt_grads_x - pred_grads_x
            diff_y = gt_grads_y - pred_grads_y

            return tf.reduce_mean(tf.abs(diff_x)) + tf.reduce_mean(tf.abs(diff_y))

        loss_option = self.params.SINGLEVIEW_LOSS
        if loss_option == 'mse':
            total_loss = mse_loss()
        elif loss_option == 'mae':
            total_loss = mae_loss()
        elif loss_option == 'scaleInv':
            total_loss = scaleInv_loss()
        elif loss_option == 'mse_mix_grad':
            total_loss = mse_loss() + gradient_loss()
        elif loss_option == 'scaleInv_mix_grad':
            total_loss = scaleInv_loss() + gradient_loss()
        else:
            pass

        return total_loss

    def semantic_loss(self, pred, gt):
        # get annotation label batch
        annotation_batch_labels = utils.get_annotation_from_label_batch(gt)

        # get valid logits and labels, two tensors of size (num_valid_eintries, NUM_CATEGORIES)
        valid_gt_tensor, valid_pred_out = utils.get_valid_logits_and_labels(
            annotation_batch_tensor=annotation_batch_labels,
            logits_batch_tensor=pred,
            class_labels=[v for v in range(self.params.NUM_CATEGORIES + 1)])

        total_loss = None

        def cross_entropy_loss():
            cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_pred_out,
                                                                         labels=valid_gt_tensor)
            cross_entropy_sum = tf.reduce_mean(cross_entropies)
            return cross_entropy_sum

        loss_option = self.params.SEMANTIC_LOSS
        if loss_option == 'categorical_crossentropy':
            total_loss = cross_entropy_loss()
        else:
            pass

        return total_loss
