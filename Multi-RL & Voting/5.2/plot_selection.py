import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

files=['./voting_test_result/real votingselected.csv','./voting_test_result/test votingselected.csv']

pplabel1=['Rain1','Rain2','Rain3','Rain4',
          'Rain5','Rain6','Rain7','Rain8']

data=[]
for name in files:
    df=pd.read_csv(name).values
    tem=[]
    for i in range(df.shape[0]):
        line=[]
        for j in range(df.shape[1]):
            if df[i][j]=='dqn':
                line.append(1)
            elif df[i][j]=='ddqn':
                line.append(2)
            elif df[i][j]=='ppo1':
                line.append(3)
            elif df[i][j]=='ppo2':
                line.append(4)
            else:
                line.append(5)
        tem.append(line)
    data.append(np.array(tem))
    
tdata=np.concatenate((data[0],data[1]),axis=1)
print(tdata.shape)


#画4张图，每张代表一个测试中选取的降雨    

font0 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

fig = plt.figure(figsize=(20,16))

for it in range(8):
    dd=tdata[:,it]
    
    x=np.arange(dd.shape[0])
    
    fig.add_subplot(4,2,it+1)
    plt.scatter(x,dd,marker='o',linewidths=1)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.xticks([0,dd.shape[0]],['8:00','16:00'],fontsize=10)
    plt.title(pplabel1[it],fontdict=font1)
    
    plt.yticks([1,2,3,4,5],['DQN','DDQN','PPO1','PPO2','A2C'],fontsize=10)

#plt.text(-320,4,'With QVI', fontdict=font0,rotation=90)
#plt.text(-320,15,'Without QVI', fontdict=font0,rotation=90)
fig.savefig('_6.3.png', bbox_inches='tight', dpi=500)  
