import numpy as np
import tensorflow as tf
#import gym
import pandas as pd

import gym


class Actor(object):                    #Policy net
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        
        self.sess = sess

        self.s = tf.compat.v1.placeholder(tf.float32, [None, n_features], "state")   #state input
        self.a = tf.compat.v1.placeholder(tf.int32, None, "act")                  #action input
        self.td_error = tf.compat.v1.placeholder(tf.float32, None, "td_error")    # TD_error

        with tf.compat.v1.variable_scope('Actor_a2c',reuse=tf.compat.v1.AUTO_REUSE):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.compat.v1.constant_initializer(0.1),  # biases
                name='l1',
                reuse=tf.compat.v1.AUTO_REUSE
            )                   #fully connected layer 1

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,  #get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.compat.v1.constant_initializer(1.),  # biases
                name='acts_prob',
                reuse=tf.compat.v1.AUTO_REUSE
            )               #output softmax

        with tf.compat.v1.variable_scope('exp_v',reuse=tf.compat.v1.AUTO_REUSE):
            log_prob = tf.math.log(self.acts_prob[0, self.a])        #selecte the action prob value
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.compat.v1.variable_scope('train',reuse=tf.compat.v1.AUTO_REUSE):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)
            

    def learn(self, s, a, td):
        s = s[np.newaxis, :].reshape((-1,5))
        feed_dict = {self.s: s, self.a: a, self.td_error: td}  #td temproal difference ??? critic net??????
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)  #??????
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        tp=probs.ravel()
        if np.isnan(tp[0]):
            probs=np.array([[1]])
            #print(probs)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int ???????????????????????????


class Critic(object):
    def __init__(self, sess, n_features, GAMMA, lr=0.01):
        self.sess = sess

        self.s = tf.compat.v1.placeholder(tf.float32, [None, n_features], "state")   #critic net input ????????????
        self.v_ = tf.compat.v1.placeholder(tf.float32, [None, 1], "v_next")         #?????????????????????value ???
        self.r = tf.compat.v1.placeholder(tf.float32, None, 'r')                  #???????????????????????????????????????
        self.GAMMA=GAMMA
        with tf.compat.v1.variable_scope('Critic_a2c',reuse=tf.compat.v1.AUTO_REUSE):                           #??????critic ??????????????????????????????????????????value
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.compat.v1.variable_scope('squared_TD_error',reuse=tf.compat.v1.AUTO_REUSE):
            self.td_error = self.r + GAMMA * self.v_ - self.v   #????????????????????? ???TD-error
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.compat.v1.variable_scope('train',reuse=tf.compat.v1.AUTO_REUSE):
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):      #critic net???????????????
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})    #????????????????????????????????????????????????q???
        td_error, _ = self.sess.run([self.td_error, self.train_op],    
                                          {self.s: s, self.v_: v_, self.r: r})  #???????????????????????????????????????q???
        return td_error




class A2C(object):
    
    def __init__(self,MAX_EPISODE,MAX_EP_STEPS,num_rain,env,raindata):
        np.random.seed(2)
        self.action_table=pd.read_excel('./action_table_of_DQN.xlsx').values[:,1:]
        GAMMA = 0.9     # reward discount in TD error
        LR_A = 0.001    # learning rate for actor
        LR_C = 0.001     # learning rate for critic
        
        # Superparameters
        self.MAX_EPISODE = MAX_EPISODE
        self.MAX_EP_STEPS = MAX_EP_STEPS   # maximum time step in one episode
        self.GAMMA = GAMMA     # reward discount in TD error
        self.LR_A = LR_A    # learning rate for actor
        self.LR_C = LR_C     # learning rate for critic
        
        self.env = env#gym.make('MountainCar-v0')
        self.t='a2c'
        self.N_F = env.observation_space#.shape[0]   #?????????????????????
        self.N_A = self.action_table.shape[0]               #?????????????????????
        self.num_rain=num_rain
        
        
        self.sess = tf.compat.v1.Session()
        self.actor = Actor(self.sess, n_features=self.N_F, n_actions=self.N_A, lr=self.LR_A)
        self.critic = Critic(self.sess, n_features=self.N_F, GAMMA=self.GAMMA, lr=self.LR_C)
        self.sess.run(tf.compat.v1.global_variables_initializer())   
        
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#????????????????????????
        self.raindata=raindata
        if raindata=='test':
            self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#????????????????????????
        else:
            self.testRainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*20#????????????????????????
        self.rainnum,m=self.rainData.shape
        
        self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
        
        

    def train(self,save):
        for j in range(self.MAX_EPISODE):     
            states1, states2, actions, track_r = [], [], [], []
            for i in range(self.num_rain):
                print('Steps: ',j,' Rain: ',i)
                s,_ =self.env.reset(self.rainData[i],i,True)
                t = 0                
                while True:
                    ap = self.actor.choose_action(s)
                    a = self.action_table[ap,:].tolist()
                    s_, r, done, info,_ = self.env.step(a,self.rainData[i])
                    track_r.append(r)
                    states1.append(s)
                    states2.append(s_)
                    actions.append(ap)
                    s = s_
                    t += 1
                    
                    if done:
                        break

            td_error = self.critic.learn(np.array(states1)[0], np.array(track_r)[0], np.array(states2)[0])  # gradient = grad[r + gamma * V(s_) - V(s)]
            self.actor.learn(np.array(states1)[0], np.array(actions)[0], td_error)     # true_gradient = grad[logPi(s,a) * td_error]
            #????????????
            if save:
                saver=tf.compat.v1.train.Saver()
                sp=saver.save(self.sess,'./'+self.t+'_test_result/model/'+self.t+'_model.ckpt')
                print("model saved:",sp)

    def load_model(self):
        saver=tf.compat.v1.train.Saver()
        saver.restore(self.sess,'./'+self.t+'_test_result/model/'+self.t+'_model.ckpt')


    def test(self,test_num):

        flooding_logs,hc_flooding_logs=[],[]
        for i in range(test_num):
            print('test'+str(i))
            s,flooding =self.env.reset(self.testRainData[i],i,False)
            
            hc_name='./'+self.t+'_test_result/HC/HC'+str(i)
            self.env.copy_result(hc_name+'.inp',self.env.orf_rain+'.inp')
            hc_flooding = self.env.reset_HC(hc_name)
            
            flooding_log,hc_flooding_log=[flooding],[hc_flooding]
            
            t = 0
            track_r = []
            while True:
                ap = self.actor.choose_action(s)
                a = self.action_table[ap,:].tolist()
                s_, r, done, info, flooding = self.env.step(a,self.testRainData[i])
                
                #??????HC,?????????HC????????????flooding
                _, hc_flooding = self.env.step_HC(hc_name)
                flooding_log.append(flooding)
                hc_flooding_log.append(hc_flooding)
        
                track_r.append(r)
                
                s = s_
                t += 1
                if done:
                    dr.append(track_r)
                    break
                
            
            #?????????????????????????????????flooding?????????
            flooding_logs.append(flooding_log)
            hc_flooding_logs.append(hc_flooding_log)
            #save RLC .inp and .rpt
            if self.raindata=='test':
                k=0
            else:
                k=4
            sout='./'+self.t+'_test_result/'+str(i+k)+'.rpt'
            sin=self.env.staf+'.rpt'
            self.env.copy_result(sout,sin)
            sout='./'+self.t+'_test_result/'+str(i+k)+'.inp'
            sin=self.env.staf+'.inp'
            self.env.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)#?????????????????????flooding?????????
            df = pd.DataFrame(np.array(flooding_logs).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
            df = pd.DataFrame(np.array(hc_flooding_logs).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
