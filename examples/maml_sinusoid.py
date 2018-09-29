# Copyright 2018 Xinyang Geng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Model Agnostic Meta Learning [https://arxiv.org/pdf/1703.03400.pdf] for sinusoid
regression. This example further demonstrates the purpose of the parameterized
model library
"""

import numpy as np
import tensorflow as tf
from parameterized_model import ParameterizedModel


def fc_net(input_tensor):
    """Simple function for constructing a fully connected network."""
    x = input_tensor
    
    for hidden_dim in [40, 40]:
        x = tf.layers.dense(x, units=hidden_dim)
        x = tf.nn.relu(x)
        
    return tf.layers.dense(x, units=1)


def main():
    # Hyperparameter configurations.
    NUM_TASKS = 10
    ADAPTATION_BATCH_SIZE = 10
    TEST_BATCH_SIZE = 256
    INNER_LOOP_LR = 0.01
    OUTER_LOOP_LR = 0.003
    NUM_TRAIN_STEPS = 10000
    
    
    # Construct a ParameterizedModel object with name.
    pm = ParameterizedModel(name='maml_sinusoid')
    
    # Allow the ParameterizedModel library to analyze the model by constructing
    # the network once. After the analysis, the library will build the single
    # one dimensional parameter vector. The initialization will be the same as
    # specified during the template construction.
    with pm.build_template():
        fc_net(tf.zeros([ADAPTATION_BATCH_SIZE, 1]))
        

    inner_loop_losses = []
    outer_loop_losses = []
    
    # Loop over different meta tasks.
    for _ in range(NUM_TASKS):
        
        # Sample the task.
        aplitude = tf.random_uniform(shape=[], minval=0.1, maxval=5.0)
        phase = tf.random_uniform(shape=[], minval=0.0, maxval=np.pi)
        
        # Construct adapation (inner loop) data.
        adaptation_x = tf.random_uniform(
            shape=[ADAPTATION_BATCH_SIZE, 1], minval=-5.0, maxval=5.0
        )
        adaptation_y = aplitude * tf.sin(adaptation_x + phase)
        
        # Forward the parameterized network with its own parameter.
        with pm.build_parameterized():
            adaptation_predicted_y = fc_net(adaptation_x)
            
        inner_loop_loss = tf.reduce_mean(
            tf.square(adaptation_predicted_y - adaptation_y)
        )
        inner_loop_losses.append(inner_loop_loss)
        
        # Compute the inner loop gradient. Since all weights are contained in
        # the single parameter vector "pm.parameter", the gradient is also a
        # single vector.
        inner_loop_gradient = tf.gradients(inner_loop_loss, pm.parameter)[0]
        
        # One step gradient descent.
        updated_parameter = pm.parameter - INNER_LOOP_LR * inner_loop_gradient
        
        
        # Construct test (outer loop) data.
        test_x = tf.random_uniform(
            shape=[TEST_BATCH_SIZE, 1], minval=-5.0, maxval=5.0
        )
        test_y = aplitude * tf.sin(test_x + phase)
        
        
        # Forward the parameterized network with updated parameter. The library
        # will take care of correctly substituting the weights with provided
        # tensor.
        with pm.build_parameterized(updated_parameter):
            test_predicted_y = fc_net(test_x)
            
        outer_loop_loss = tf.reduce_mean(
            tf.square(test_predicted_y - test_y)
        )
        
        outer_loop_losses.append(outer_loop_loss)
        
    mean_inner_loop_loss = tf.add_n(inner_loop_losses) / len(inner_loop_losses)
    mean_outer_loop_loss = tf.add_n(outer_loop_losses) / len(outer_loop_losses)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=OUTER_LOOP_LR)
    train_op = optimizer.minimize(mean_outer_loop_loss)
    
    init_op = tf.global_variables_initializer()
    
    
    with tf.Session() as sess:
        sess.run(init_op)
        
        # The main training loop.
        for step in range(NUM_TRAIN_STEPS):
            inner_loss_val, outer_loss_val, _ = sess.run(
                [mean_inner_loop_loss, mean_outer_loop_loss, train_op]
            )
            
            if step % 100 == 0:
                print(
                    'Step: {}, inner loop loss: {}, outer loop loss: {}'.format(
                        step, inner_loss_val, outer_loss_val
                    )
                )
    
if __name__ == '__main__':
    main()
