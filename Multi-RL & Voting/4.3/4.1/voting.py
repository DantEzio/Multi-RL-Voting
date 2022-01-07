import A2C
import PPO
import DDQN
import env_SWMM
import numpy as np
import pandas as pd
import tensorflow as tf
import get_rpt
import get_output


class Voting():
    def __init__(self,MAX_EPISODE,MAX_EP_STEPS,env,num_rain,raindata):
        
        test_num=4
        g_1 = tf.Graph()
        with g_1.as_default():
            self.model_dqn = DDQN.DDQN(step=MAX_EPISODE,batch_size=MAX_EP_STEPS,num_rain=num_rain,env=env,t='dqn',raindata=raindata)
            self.model_dqn.load_model()
            #self.model_dqn.test(test_num)
            print('dqn')
            
        g_2 = tf.Graph()
        with g_2.as_default():
            self.model_ddqn = DDQN.DDQN(step=MAX_EPISODE,batch_size=MAX_EP_STEPS,num_rain=num_rain,env=env,t='ddqn',raindata=raindata)
            self.model_ddqn.load_model()
            #self.model_ddqn.test(test_num)
            print('ddqn')
            
        g_3 = tf.Graph()
        with g_3.as_default():
            self.model_ppo1 = PPO.PPO(env, MAX_EPISODE,MAX_EP_STEPS,num_rain, 'ppo1',raindata)
            self.model_ppo1.load_model()
            #self.model_ppo1.test(test_num)
            print('ppo1')
            
        g_4 = tf.Graph()
        with g_4.as_default():
            self.model_ppo2 = PPO.PPO(env, MAX_EPISODE, MAX_EP_STEPS,num_rain,'ppo2',raindata)
            self.model_ppo2.load_model()
            #self.model_ppo2.test(test_num)
            print('ppo2')
            
        g_5 = tf.Graph()
        with g_5.as_default():
            self.model_a2c = A2C.A2C(MAX_EPISODE,MAX_EP_STEPS,num_rain,env,raindata)
            self.model_a2c.load_model()
            #self.model_a2c.test(test_num)
            print('a2c')

        self.t='voting'
        
        self.rainData=np.loadtxt('./sim/trainRainFile.txt',delimiter=',')#读取训练降雨数据
        self.raindata=raindata
        if raindata=='test':
            self.testRainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
        else:
            self.testRainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*20#读取真实降雨数据
        self.rainnum,m=self.rainData.shape
        
        
        self.rainData=np.hstack((self.rainData,np.zeros((self.rainnum,m))))
        self.testRainData=np.hstack((self.testRainData,np.zeros((self.testRainData.shape[0],m))))
    
         
    def test(self,test_num,env):

        dr=[]
        Ns=['WS02006235','WS02006252']
        self.action_table=pd.read_excel('./action_table_of_DQN.xlsx').values[:,1:]
        
        flooding_logs,hc_flooding_logs=[],[]
        select_log=[]
        for i in range(test_num):
            print('test'+str(i))
            
            observation,flooding = env.reset(self.testRainData[i],i,False)
            #准备optimization of AFI使用的inp文件
            #change_rain.copy_result(env.staf+'.inp',AFI_env+'.inp')
            #用于对比的HC,HC有RLC同步，共用一个iten计数器，
            #所以HC的reset要紧跟RLC的reset，HC的step要紧跟RLC的step，保证iten变量同步
            
            flooding_log=[flooding]
            selects=[]
            states, actions, rewards = [], [], []
            while True:
                a1 = self.model_ppo1.choose_action(observation)
                a2 = self.model_ppo2.choose_action(observation)
                a = self.model_dqn.choose_action(observation)
                a3 = self.action_table[a,:].tolist()
                a = self.model_ddqn.choose_action(observation)
                a4 = self.action_table[a,:].tolist()
                a = self.model_a2c.actor.choose_action(observation)
                a5 = self.action_table[a,:].tolist()
                
                atem=[a1,a2,a3,a4,a5]
                wl=[]
                floodtem=[]
                for j in range(len(atem)):
                    _,_,_,_,flooding = env.step(atem[j],self.testRainData[i])
                    floodtem.append(flooding)
                    #Get water level of N1 and N2
                    wlDic=get_output.depth(env.staf+'.out',Ns,env.iten)
                    temw=0
                    for key in Ns:
                        temw+=wlDic[key]
                    wl.append(temw/len(Ns))
                    env.iten-=1
                    env.action_seq.pop()
                
                #print(wl)
                picksf=[i for i, x in enumerate(floodtem) if x == np.min(floodtem)]
                picksl=wl.index(min(wl))
                if len(picksf)>1:
                    if picksl in picksf:
                        pick=picksl
                    else:
                        pick=picksf[0]
                else:
                    pick=picksl
                #print(picksf,picksl,pick)
                
                if pick==0:
                    selects.append('ppo1')
                elif pick==1:
                    selects.append('ppo2')
                elif pick==2:
                    selects.append('dqn')
                elif pick==3:
                    selects.append('ddqn')
                else:
                    selects.append('a2c')
                a=atem[pick]
                next_observation, reward, done, _, flooding = env.step(a,self.testRainData[i])
                
                #对比HC,也记录HC每一步的flooding
                
                states.append(observation)
                actions.append(a)
                
                flooding_log.append(flooding)
                
                rewards.append(reward)

                observation = next_observation
                    
                if done:
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    d_reward = self.model_ppo1.discount_reward(states, rewards, next_observation)

                    states, actions, rewards = [], [], []
                    dr.append(d_reward)
                    
                    break
            #一场降雨结束后记录一次flooding过程线
            flooding_logs.append(flooding_log)
            select_log.append(selects)
            
            #save RLC .inp and .rpt
            if self.raindata=='test':
                k=0
            else:
                k=4
            
            sout='./'+self.t+'_test_result/'+str(i+k)+'.rpt'
            sin=env.staf+'.rpt'
            env.copy_result(sout,sin)
            sout='./'+self.t+'_test_result/'+str(i+k)+'.inp'
            sin=env.staf+'.inp'
            env.copy_result(sout,sin)
            #self.env.copy_result(sout,sin)#保存所有降雨的flooding过程线
            df = pd.DataFrame(np.array(flooding_logs).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'flooding_vs_t.csv', index=False, encoding='utf-8')
            df = pd.DataFrame(np.array(hc_flooding_logs).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'hc_flooding_vs_t.csv', index=False, encoding='utf-8')
            df = pd.DataFrame(np.array(select_log).T)
            df.to_csv('./'+self.t+'_test_result/'+self.raindata+' '+self.t+'selected.csv', index=False, encoding='utf-8')
            
        return dr
    
if __name__ == '__main__':
    
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00']
    
    date_t=[]
    for i in range(len(date_time)):
        date_t.append(int(i*10))
    
    num_rain=4
    AFI=False
    raindata='test'#'real'
    testnum=4
    
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    model=Voting(10,100,env,num_rain,'test') 
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    model.test(testnum,env)
    
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    model=Voting(2,100,env,num_rain,'real') 
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    model.test(testnum,env)
