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
Example of computing the full Hessian matrix of a small neural network.
"""


import tensorflow as tf
from parameterized_model import ParameterizedModel


def fc_net(input_tensor, output_dim):
    """Simple function for constructing a fully connected network."""
    x = input_tensor
    
    for hidden_dim in [32, 32]:
        x = tf.layers.dense(x, units=hidden_dim)
        x = tf.nn.relu(x)
        
    return tf.layers.dense(x, units=output_dim)
    

def main():
    BATCH_SIZE = 16
    INPUT_DIM = 8
    OUTPUT_DIM = 8
    
    input_tensor = tf.random_normal(shape=[BATCH_SIZE, INPUT_DIM], dtype=tf.float32)
    ground_truth = tf.random_normal(shape=[BATCH_SIZE, OUTPUT_DIM], dtype=tf.float32)
    
    # Construct a ParameterizedModel object with name.
    pm = ParameterizedModel(name='fully_connected_network')
    
    
    # Allow the ParameterizedModel library to analyze the model by constructing
    # the network once. After the analysis, the library will build the single
    # one dimensional parameter vector. The initialization will be the same as
    # specified during the template construction.
    with pm.build_template():
        fc_net(input_tensor, OUTPUT_DIM)

    
    # Construct the parameterized network.
    with pm.build_parameterized():
        output_tensor = fc_net(input_tensor, OUTPUT_DIM)
        
        
    loss = tf.reduce_mean(tf.nn.l2_loss(output_tensor - ground_truth))
    
    # pm.parameter is now the single vector variable for the model.
    print('Parameter size: {}'.format(pm.parameter.shape))
    
    # Construct the graph for the Hessian.
    hessian = tf.hessians(loss, pm.parameter)[0]
    
    
    variable_intializaer = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(variable_intializaer)
    
        # Compute the Hessian and validate the shape    
        hessian_value = sess.run(hessian)
        print('Hessian shape: {}'.format(hessian_value.shape))


if __name__ == '__main__':
    main()
