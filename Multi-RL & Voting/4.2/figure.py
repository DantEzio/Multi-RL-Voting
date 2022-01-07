import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


nums=['0','1','2','3','4','5','6','7']
nms=['dqn_test_result','ddqn_test_result','ppo1_test_result',
     'ppo2_test_result','a2c_test_result','voting_test_result']
ps=['1','2']


font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 8,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 15,}


#N1
N=str(1)
labels=['mean','median']
fig = plt.figure(figsize=(15,15))
for lab in range(len(labels)):
    
    tem_mean=pd.read_excel('./water level_N'+N+'_'+labels[lab]+'_.xlsx',sheet_name=labels[lab]).values 
    print(tem_mean.shape)
    name=tem_mean[:,1]
    #print(name)
    print(tem_mean[0,1:])
         
    plts=fig.add_subplot(3,1,lab+1)
        
    bar_width = 0.1 # 条形宽度
    index_dqn = np.arange(len(tem_mean[1,1:]))
    index_ddqn = index_dqn + bar_width 
    index_ppo1 = index_ddqn + bar_width 
    index_ppo2 = index_ppo1 + bar_width 
    index_a2c = index_ppo2 + bar_width 
    index_voting = index_a2c + bar_width 
        
    plt.bar(index_dqn,height=tem_mean[0,1:],width=bar_width,color='g',label='DQN')
    plt.bar(index_ddqn,height=tem_mean[1,1:],width=bar_width,color='r',label='DDQN')
    plt.bar(index_ppo1,height=tem_mean[2,1:],width=bar_width,color='b',label='PPO1')
    plt.bar(index_ppo2,height=tem_mean[3,1:],width=bar_width,color='c',label='PPO2')
    plt.bar(index_a2c,height=tem_mean[4,1:],width=bar_width,color='m',label='A2C')
    plt.bar(index_voting,height=tem_mean[5,1:],width=bar_width,color='k',label='Voting')
        
    #plt.title('N'+p,font2)
    if lab==0:
        st='Mean value of Water levels (m)'
    elif lab==1:
        st='Median value of Water levels (m)'
    else:
        st='Variance value of Water levels (m)'
    
    plt.ylabel(st,font2)
    plt.xticks([0.2,1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2],['Rain1','Rain2','Rain3','Rain4','Rain5','Rain6','Rain7','Rain8',' '])

    plt.legend(prop=font1)

       
fig.savefig('5.3.'+N+'.png', bbox_inches='tight', dpi=1000)

