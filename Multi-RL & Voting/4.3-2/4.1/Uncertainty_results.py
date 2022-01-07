import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def handle_line(line, flag, title):
    if line.find(title) >= 0:
        flag = True
    elif flag and line == "":
        flag = False
    return flag

def get_rpt(filename):
    with open(filename, 'rt') as data:
        total_in=0
        flooding=0
        pumps_flag = outfall_flag= False
        for line in data:
            # Aim at three property to update origin data
            line = line.rstrip('\n')
            #pumps_flag = handle_line(line, pumps_flag, 'Quality Routing Continuity')
            pumps_flag = handle_line(line, pumps_flag, 'Flow Routing Continuity')
            outfall_flag=handle_line(line, outfall_flag, 'Outfall Loading Summary')
            node = line.split() # Split the line by whitespace
            if pumps_flag and node!=[]:
                if line.find('Flooding Loss')>=0:
                    flooding=float(node[3])
                elif line.find('Dry Weather Inflow')>=0 or \
                     line.find('Wet Weather Inflow')>=0 :
                    total_in+=float(node[4])                  
                elif line.find('Groundwater Inflow')>=0 or \
                     line.find('RDII Inflow')>=0 or \
                     line.find('External Inflow')>=0:
                    total_in+=float(node[3])
    return total_in,flooding


#获取文件名
file_name=['./dqn_test_result/','./ddqn_test_result/',
           './ppo1_test_result/','./ppo2_test_result/',
           './a2c_test_result/','./voting_test_result/']


Ps=[i for i in range(1,51)]
x=[]
for p in range(50):
    x.append(float(Ps[int(20*p/50)]))


font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 10,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

colors=['r','g','b','c','m','k']
labels=[['DQN'],['DDQN'],['PPO1'],['PPO2'],['A2C'],['Voting']]
it=0


fig = plt.figure(figsize=(15,15))
for name in file_name:
    #plts=fig.add_subplot(3,2,it+1)
    #plts.figsize=(5, 5)
    all_rate=[]
    
    for i in range(4):
        for j in range(40):
            f=name+str(i)+'_'+str(j)+'.rpt'
            total_in,CSO=get_rpt(f)#读取total_inflow和CSO
            all_rate.append(CSO/total_in)#计算比值RIC
         
    
    #找到5%，50%，95%的结果
    all_rate.sort()
    print(name+':')
    print('5%:',all_rate[int(50*5/100)],' ',
          '50%:',all_rate[int(50*50/100)],' ',
          '95%:',all_rate[int(50*95/100)])
    
    
    '''
    #画6.5的图
    plt.scatter(x,all_rate,c=colors[it])
    #plt.title(labels[it],fontdict=font2)
    
    # 增加标签
    plt.xlabel('P',fontdict=font1)
    plt.ylabel('RCI',fontdict=font1)
    
    # 增加刻度
    plt.xticks(x, x)
    
    # 设置图例
    n=labels[it]
    print(n)
    plt.legend(n,loc='best',prop=font2)
    
    k=linregress(x,all_rate)[0]
    b=linregress(x,all_rate)[1]
    print(k,b)
    x1=np.array(x)
    mean=k*x1+b
    plt.plot(x,mean,color=colors[it],lw=1.5,ls='--',zorder=2)
    
    it=it+1
fig.savefig('5.1.1.png', bbox_inches='tight', dpi=500)   
'''