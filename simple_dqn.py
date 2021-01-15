# Credit to: https://www.youtube.com/watch?v=wYIiMH1cIis&t=447s

import os
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np

#print(os.path)
class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, input_dims,
                  fc1_dims = 256, fc2_dims=256, chkpt_dir='./checkpoint'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, "2hu.ckpt")
        
    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name="inputs")
            self.actions = tf.placeholder(tf.float32,
                                           shape=[None, self.n_actions],
                                           name="actions_taken")
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, self.n_actions],
                                           name="q_value")
            flat = tf.layers.flatten(self.input)
            dense1 = tf.layers.dense(flat, units=self.fc1_dims,
                                    activation = tf.nn.relu)
            dense2 = tf.layers.dense(dense1, units=self.fc2_dims,
                                     activation=tf.nn.relu)
            self.Q_values = tf.layers.dense(dense2, self.n_actions)
            self.loss = tf.reduce_mean(tf.square(self.Q_values-self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            
    def load_checkpoint(self):
        #print(".... loading checkpoint ....")
        self.saver.restore(self.sess, self.checkpoint_file)
    
    def save_checkpoint(self):
        self.saver.save(self.sess, self.checkpoint_file)

class Agent(object):
    def __init__(self, lr, gamma, mem_size, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=0.9996, epsilon_end=0.01, q_dir="./checkpoint"):
            self.action_space = [i for i in range(n_actions)]
            self.n_actions = n_actions
            self.gamma = gamma
            self.mem_size = mem_size
            self.mem_cntr = 0
            self.epsilon = epsilon
            self.epsilon_dec = epsilon_dec
            self.epsilon_end = epsilon_end
            self.batch_size = batch_size
            self.q_eval = DeepQNetwork(lr, n_actions, input_dims=input_dims,
                                        name="q_eval", chkpt_dir=q_dir)
            self.state_memory = np.zeros((self.mem_size, *input_dims))
            self.new_state_memory = np.zeros((self.mem_size, *input_dims))
            self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                            dtype=np.int8)
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
        
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        #print("mem_size: ", self.mem_size)
        print("index stored at: ", index)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.terminal_memory[index] = 1  -terminal
        self.mem_cntr += 1
        
    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                         feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        return action
     
    def learn(self):
        if self.mem_cntr > self.batch_size:
            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                    else self.mem_size
            
            batch = np.random.choice(max_mem, self.batch_size)
            
            state_batch = self.state_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            terminal_batch = self.terminal_memory[batch]
            
            q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                        feed_dict={self.q_eval.input: state_batch})
            
            q_next = self.q_eval.sess.run(self.q_eval.Q_values,
                                    feed_dict = {self.q_eval.input: new_state_batch})
            
            q_target = q_eval.copy()
            
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward_batch + \
                                            self.gamma*np.max(q_next, axis=1)*terminal_batch
            
            _ = self.q_eval.sess.run(self.q_eval.train_op,
                                    feed_dict={self.q_eval.input: state_batch,
                                               self.q_eval.actions: action_batch,
                                               self.q_eval.q_target: q_target})
            
            #writer = tf.summary.FileWriter('./graphs', self.q_eval.sess.graph)
            
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                            self.epsilon_end else self.epsilon_end
                            
            print("EPSILON is: ", self.epsilon)
                            
    def save_models(self):
        self.q_eval.save_checkpoint()
        
    def load_models(self):
        self.q_eval.load_checkpoint()
    
        