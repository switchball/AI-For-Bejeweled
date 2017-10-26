#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File SpriteConvnetModel.py created on 9:33 2017/9/19 

@author: Yichi Xiao
@version: 1.0
"""

import numpy as np
import tensorflow as tf
import time
import os

class SpriteConvnetModel:
    def __init__(self, config, reuse, is_training):
        self.sess = None
        self.graph = None
        self.open()
        self.variable_average = None
        self.LABEL_NUM = 23

        #with self.graph.as_default(), self.sess as sess:
        self.get_network(reuse, is_training)

        self.sess = tf.Session(graph=self.predictions.graph)
        self.sess.run(tf.global_variables_initializer())

        self.config = config
        self.config.learning_base = 1e-4

        # 0      -> null
        # 1-7    -> normal
        # 9->15  -> fire (+8)
        # 16->22 -> cross (+15)
        # 8     -> special

    def get_network(self, reuse=False, is_training=False):
        """Model function for CNN."""
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # MNIST images are 28x28 pixels, and have one color channel
        # input_layer = tf.reshape(features, [-1, 32, 32, 3])
        input_layer = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels = tf.placeholder(tf.int32, [None])
        self.x_input = input_layer
        self.y_label = labels

        with tf.variable_scope("convnet", reuse=reuse):
            a, b = None, None
            with tf.variable_scope("conv1"):
                # Convolutional Layer #1
                # Computes 8 features using a 5x5 filter with ReLU activation.
                # Padding is added to preserve width and height.
                # Input Tensor Shape: [batch_size, 32, 32, 3]
                # Output Tensor Shape: [batch_size, 28, 28, 24]
                print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
                conv1 = tf.layers.conv2d(
                        inputs=input_layer,
                        filters=8,
                        kernel_size=[5, 5],
                        padding="valid",
                        activation=tf.nn.relu)
                print(conv1.name)

            with tf.variable_scope("pool1"):
                # Pooling Layer #1
                # First max pooling layer with a 2x2 filter and stride of 2
                # Input Tensor Shape: [batch_size, 28, 28, 24]
                # Output Tensor Shape: [batch_size, 14, 14, 24]
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            with tf.variable_scope("conv2"):
                # Convolutional Layer #2
                # Computes 64 features using a 5x5 filter.
                # Padding is added to preserve width and height.
                # Input Tensor Shape: [batch_size, 14, 14, 12]
                # Output Tensor Shape: [batch_size, 10, 10, 64]
                conv2 = tf.layers.conv2d(
                        inputs=pool1,
                        filters=16,
                        kernel_size=[5, 5],
                        padding="valid",
                        activation=tf.nn.relu)

            with tf.variable_scope("pool2"):
                # Pooling Layer #2
                # Second max pooling layer with a 2x2 filter and stride of 2
                # Input Tensor Shape: [batch_size, 10, 10, 64]
                # Output Tensor Shape: [batch_size, 5, 5, 64]
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

                # Flatten tensor into a batch of vectors
                # Input Tensor Shape: [batch_size, 5, 5, 64]
                # Output Tensor Shape: [batch_size, 5 * 5 * 64]
                pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

            with tf.variable_scope("fc"):
                # Dense Layer
                # Densely connected layer with 256 neurons
                # Input Tensor Shape: [batch_size, 7 * 7 * 64]
                # Output Tensor Shape: [batch_size, 128]
                dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

            with tf.variable_scope("dropout"):
                # Add dropout operation; 0.6 probability that element will be kept
               dropout = tf.layers.dropout(
                       inputs=dense, rate=0.4, training=is_training)

            with tf.variable_scope("logits"):
                # Logits layer
                # Input Tensor Shape: [batch_size, 256]
                # Output Tensor Shape: [batch_size, 8]
                logits = tf.layers.dense(inputs=dense, units=self.LABEL_NUM)

            with tf.variable_scope("predictions"):
                probabilities = tf.nn.softmax(logits, name="softmax_tensor")
                # Generate predictions (for PREDICT and EVAL mode)
                predictions = tf.argmax(input=logits, axis=1)

            # Calculate Loss (for both TRAIN and EVAL modes)
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.LABEL_NUM)
            loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels, logits=logits)

            # Add evaluation metrics (for EVAL mode)
            eval_metric_ops = {
                    "accuracy": tf.metrics.accuracy(
                            labels=labels, predictions=predictions)}

            # Configure the Training Op (for TRAIN mode)
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001)
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

        self.logits = logits
        self.predictions = predictions
        self.loss = loss
        self.train_op = train_op
        self.accuracy = eval_metric_ops["accuracy"]

    def open(self):
        if self.graph is None:
            self.graph = tf.Graph()
        if self.sess is None or self.sess:
            self.sess = tf.Session(graph=self.graph)
        return self.sess

    def close(self):
        self.sess.close()

    def train_prepare(self, restore_model):
        sess = self.open()
        with sess:
            FLAGS = tf.app.flags
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.maximum(1e-7, tf.train.exponential_decay(self.config.learning_base, global_step,
                                                                        self.config.decay_step, self.config.learning_rate_decay,
                                                                        staircase=True))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
            apply_gradient_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
            variable_averages = tf.train.ExponentialMovingAverage(
                self.config.moving_average_decay, global_step)
            self.variable_average = variable_averages
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                self.train_op = tf.no_op(name='train')

            grad_summaries = []
            grad_summaries.append(tf.summary.scalar('loss', self.loss))
            grad_summaries.append(tf.summary.scalar('accuracy', self.accuracy[1]))
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(tf.local_variables_initializer())
            train_writer = tf.summary.FileWriter(self.config.checkpointDir + 'train', sess.graph)

            self.learning_rate = learning_rate

            # restore model
            if restore_model:
                ckpt = tf.train.get_checkpoint_state(self.config.checkpointDir)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("[Train Prepare] Model " + ckpt.model_checkpoint_path + " restored.")

    def train(self, features, labels):
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            global_step = [var for var in tf.global_variables() if var.op.name=="global_step"][0]

            #a, b = tf.train.shuffle_batch([features, labels], batch_size=64,
            #                              capacity=64000, min_after_dequeue=10240, enqueue_many=True)
            x = 0
            batch_loss = 0
            for t in range(3002):
                if (x+1)*64 > labels.shape[0]:
                    x = 0
                if x == 0:
                    rng_state = np.random.get_state()
                    np.random.shuffle(features)
                    np.random.set_state(rng_state)
                    np.random.shuffle(labels)
                    print(int(t/86), batch_loss/86)
                    batch_loss = 0
                a = features[x*64: (x+1)*64]
                b = labels[x*64: (x+1)*64]
                _, loss = sess.run([self.train_op, self.loss], feed_dict={
                    self.x_input: a,
                    self.y_label: b
                })
                batch_loss += loss
                x+=1

                if t % 1000 == 0:
                    saver.save(sess, self.config.checkpointDir + 'model.ckpt', global_step=global_step)

        return None

    def predict_prepare(self):
        test_writer = tf.summary.FileWriter(self.config.checkpointDir + 'test', self.sess.graph)

        if self.variable_average is None:
            variable_averages = tf.train.ExponentialMovingAverage(self.config.moving_average_decay)
        else:
            variable_averages = self.variable_average
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        self.sess.run(tf.local_variables_initializer())

        # restore
        ckpt = tf.train.get_checkpoint_state(self.config.checkpointDir)
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("[Predict Prepare] Model " + ckpt.model_checkpoint_path + " restored.")


    def predict(self, features):
        preds = self.sess.run(self.predictions, feed_dict={
            self.x_input: features
        })
        return preds # self.predictions.eval({self.x_input: features})

    def predict_and_eval(self, features, labels):
        with self.open() as sess:
            sess.run(tf.global_variables_initializer())
            preds = sess.run([self.predictions, self.accuracy], feed_dict={
                self.x_input: features,
                self.y_label: labels
            })
        return preds # self.predictions.eval({self.x_input: features})


    def metric(self, predictions, labels):
        with self.open() as sess:
            acc = sess.run(self.accuracy, feed_dict={
                self.y_label: labels,
                self.predictions: predictions
            })
        return acc

def old():
    with tf.variable_scope("root"):
        # At start, the scope is not reusing.
        print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
        assert tf.get_variable_scope().reuse == False
        with tf.variable_scope("foo"):
            # Opened a sub-scope, still not reusing.
            print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
            assert tf.get_variable_scope().reuse == False
        with tf.variable_scope("foo", reuse=True):
            # Explicitly opened a reusing scope.
            print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
            assert tf.get_variable_scope().reuse == True
            with tf.variable_scope("bar"):
                # Now sub-scope inherits the reuse flag.
                print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
                assert tf.get_variable_scope().reuse == True
        # Exited the reusing scope, back to a non-reusing one.
        print(tf.get_variable_scope().name, tf.get_variable_scope().original_name_scope)
        assert tf.get_variable_scope().reuse == False

def train():
    root = 'F:/Workspace/AI-For-Bejeweled/'
    FLAGS = tf.app.flags.FLAGS
    FLAGS.learning_base = 1e-4
    FLAGS.decay_step = 10000
    FLAGS.decay_rate = 0.9
    FLAGS.max_grad_norm = 5.0
    FLAGS.moving_average_decay = 1.0
    FLAGS.checkpointDir = root + "model/new_sprite_convnet_model/"

    # Train/Test Data
    train_data = np.load(root + 'img_data/sample64.npy')
    train_data = np.asarray(train_data, dtype=np.float16)
    train_labels=np.load(root + 'img_data/label64.npy')
    train_labels=np.asarray(train_labels, dtype=np.int32)

    with tf.Graph().as_default(), tf.Session() as sess:
        start = time.clock()
        ConvModel = SpriteConvnetModel(FLAGS, False, True)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.maximum(1e-7, tf.train.exponential_decay(FLAGS.learning_base, global_step,
                                                                    FLAGS.decay_step, FLAGS.learning_rate_decay,
                                                                    staircase=True))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(ConvModel.loss, tvars), FLAGS.max_grad_norm)
        apply_gradient_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        grad_summaries = []
        grad_summaries.append(tf.summary.scalar('loss', ConvModel.loss))
        grad_summaries.append(tf.summary.scalar('accuracy', ConvModel.accuracy[0]))
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        train_writer = tf.summary.FileWriter(FLAGS.checkpointDir + 'train', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                if step % 100 == 0:
                    print(step)
                    summary, loss, acc, _ = sess.run([grad_summaries_merged, ConvModel.loss, ConvModel.accuracy, train_op], feed_dict={
                            ConvModel.x_input: train_data,
                            ConvModel.y_label: train_labels
                        })
                    train_writer.add_summary(summary, step)
                    print('Adding run metadata for', step)
                    elapsed = (time.clock() - start)
                    print("Time used:", elapsed)
                    print("loss:", loss, "acc:", acc)
                    saver.save(sess, FLAGS.checkpointDir + 'model.ckpt', global_step=global_step)
                else:
                    try:
                        _, lr = sess.run([train_op, learning_rate], feed_dict={
                            ConvModel.x_input: train_data,
                            ConvModel.y_label: train_labels
                        })
                        # print('lr:', lr)
                    except tf.errors.InvalidArgumentError:
                        print("go")
                        pass
                        # else:
                        elapsed = (time.clock() - start)
                        print("Time used:", elapsed)
                step = step + 1

        except tf.errors.OutOfRangeError:
            print(' training for 1 epochs, %d steps', step)
        finally:
            # if not os.path.exists(FLAGS.modelDir):
            #     os.mkdir(FLAGS.modelDir)
            train_writer.close()
            elapsed = (time.clock() - start)
            print("Time used:", elapsed)

            print("haha")
            coord.request_stop()
            coord.join(threads)

def predict():
    root = 'F:/Workspace/AI-For-Bejeweled/'
    FLAGS = tf.app.flags.FLAGS
    FLAGS.learning_base = 1e-4
    FLAGS.decay_step = 10000
    FLAGS.decay_rate = 0.9
    FLAGS.max_grad_norm = 5.0
    FLAGS.moving_average_decay = 1.0
    FLAGS.checkpointDir = root + "model/new_sprite_convnet_model/"

    # Train/Test Data
    train_data = np.load(root + 'img_data/sample64.npy')
    train_data = np.asarray(train_data, dtype=np.float16)
    train_labels=np.load(root + 'img_data/label64.npy')
    train_labels=np.asarray(train_labels, dtype=np.int32)

    with tf.Graph().as_default(), tf.Session() as sess:
        start = time.clock()
        _ = SpriteConvnetModel(FLAGS, reuse=False, is_training=True)
        ConvModel = SpriteConvnetModel(FLAGS, reuse=True, is_training=False)
        test_writer = tf.summary.FileWriter(FLAGS.checkpointDir + 'test', sess.graph)

        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        sess.run(tf.initialize_local_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        ckpt_old = ""
        try:
            step = 0
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpointDir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            while not coord.should_stop() and step <= 200:
                # while True:
                if ckpt and ckpt.model_checkpoint_path and ckpt_old != ckpt.model_checkpoint_path:
                    # ckpt_old = ckpt.model_checkpoint_path
                    pred = sess.run([ConvModel.predictions], feed_dict={
                        ConvModel.x_input: train_data
                    })
                    print(step, pred)
                    elapsed = (time.clock() - start)
                    print("Time used:", elapsed)
                    # test_writer.add_summary(summary, step)
                    step = step + 1
        except tf.errors.OutOfRangeError:
            print(' training for 1 epochs, %d steps', step)
        finally:
            test_writer.close()
            print("haha")
            coord.request_stop()
            coord.join(threads)



def main():
    # train()
    # predict()


    # Train/Test Data
    root = 'F:/Workspace/AI-For-Bejeweled/'
    train_data = np.load(root + 'img_data/sample64.npy')
    train_data = np.asarray(train_data, dtype=np.float16)
    train_labels=np.load(root + 'img_data/label64.npy')
    train_labels=np.asarray(train_labels, dtype=np.int32)

    root = 'F:/Workspace/AI-For-Bejeweled/'
    FLAGS = tf.app.flags.FLAGS
    FLAGS.learning_base = 1e-4
    FLAGS.decay_step = 10000
    FLAGS.decay_rate = 0.9
    FLAGS.max_grad_norm = 5.0
    FLAGS.moving_average_decay = 1.0
    FLAGS.checkpointDir = root + "model/new_sprite_convnet_model/"

    def gen_repeat():
        for _ in range(10):
            yield train_data

    from img_utils import collect_sprite_training_data
    features, labels = collect_sprite_training_data(root + "img_data")

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    model = SpriteConvnetModel(FLAGS, False, True)
    #model.train_prepare(restore_model=False)
    #model.train(features, labels)

    model.predict_prepare()
    model.predict(features)
    exit(0)

    model2= SpriteConvnetModel(FLAGS, False, False)
    gen2 = model2.predictor(gen_repeat())
    for x in gen2:
        print(x)
    print(train_labels)
    preds = model2.predict(train_data)
    print(preds)

    #preds2 = model2.predict_and_eval(train_data, train_labels)
    #print(preds2)
    #acc = model2.metric(preds, train_labels)
    #print(acc)

def tf_flags():
    tf.app.flags.DEFINE_string("tables", '', "tables info")
    tf.app.flags.DEFINE_integer("embedding_dim", -1, "Dimensionality of character embedding (default: 300)")
    tf.app.flags.DEFINE_float("margin", -1, "hash codes length")
    tf.app.flags.DEFINE_float("learning_base", 1e-3, "learning rate")
    tf.app.flags.DEFINE_float("decay_step", 10000, "decay_step")
    tf.app.flags.DEFINE_float("learning_rate_decay", 0.9, "learning_rate_decay")
    # Training parameters
    tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
    tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.app.flags.DEFINE_string("checkpointDir", '', "dir to save graph")
    tf.app.flags.DEFINE_integer("max_grad_norm", 5, "clipped")
    tf.app.flags.DEFINE_float("weight_decay", 1, "weight_decay")
    tf.app.flags.DEFINE_float("dropout", 0, "dropout")
    tf.app.flags.DEFINE_float("moving_average_decay", 1.0, "moving_average_decay")

    root = 'F:/Workspace/AI-For-Bejeweled/'
    FLAGS = tf.app.flags.FLAGS
    FLAGS.learning_base = 1e-4
    FLAGS.decay_step = 10000
    FLAGS.decay_rate = 0.9
    FLAGS.max_grad_norm = 5.0
    FLAGS.moving_average_decay = 1.0
    FLAGS.checkpointDir = root + "model/new_sprite_convnet_model/"
    return FLAGS

if __name__ == '__main__':
    # test()
    # SpriteConvnetModel()
    tf_flags()
    main()