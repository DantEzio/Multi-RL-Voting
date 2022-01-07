import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#每个df包含所有降雨结果
#获取所有测试数据
def collect_data(filename):
    data={}
    for i in range(4):
        data[i]=[]
        
    for testid in ['test']:
        for rid in range(10):
            tem=pd.read_csv('./4.1/'+filename+'_test_result/'+testid+' '+str(rid)+' '+filename+'flooding_vs_t.csv').values
            for i in range(4):
                if testid=='test':
                    k=i
                else:
                    k=i+4
                data[k].append(tem[1:,i].tolist())
    return data

dfa2c=collect_data('a2c')
dfdqn=collect_data('dqn')
dfddqn=collect_data('ddqn')
dfppo1=collect_data('ppo1')
dfppo2=collect_data('ppo2')
dfvt=collect_data('voting')

dfhc=pd.read_csv('./4.1/HC_test_result/test hcflooding_vs_t.csv').values[1:,:]
dfop=pd.read_csv('./4.1/opt_test_result/test optflooding_vs_t.csv').values[1:,:]


#datas=[dfdqn,dfddqn,dfppo1,dfppo2,dfa2c,dfvt]

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 25,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 20,}

font3 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 35,}


def draw(data,dfop,dfhc):
    a=np.max(data,axis=0)
    b=np.min(data,axis=0)    
    def func(x):
        return a[x]#+np.random.rand(1)[0]+np.random.rand(1)[0]
    #定义另一个函数
    def func1(x):
        return b[x]#+np.random.rand(1)[0]
    
    xf = [i for i in range(a.shape[0])]

    plt.fill_between(xf,func1(xf),func(xf),color='b',alpha=0.75)
    plt.plot(dfop,'k',label='Optimization model',alpha=0.5)
    plt.plot(dfhc,'k:',label='Water level system',alpha=0.5)
    #plt.legend(prop=font1)


fig = plt.figure(figsize=(20,24))
line=0
im=1
while im < 25:
    
    if line ==0:
        data=dfdqn
        ytitle='DQN'
    elif line==1:
        data=dfddqn
        ytitle='DDQN'
    elif line==2:
        data=dfppo1
        ytitle='PPO1'
    elif line==3:
        data==dfppo2
        ytitle='PPO2'
    elif line==4:
        data=dfa2c
        ytitle='A2C'
    else:
        data=dfvt
        ytitle='Voting'
    
    plts=fig.add_subplot(6,4,im)
    
    draw(data[0],dfop[:,0],dfhc[:,0])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>20:
        plt.xlabel('time',font2)
    plt.xticks([0,48],['08:00','16:00'])
    plt.ylabel(ytitle,font2)
    im+=1
    
    plts=fig.add_subplot(6,4,im)
    draw(data[1],dfop[:,1],dfhc[:,1])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>20:
        plt.xlabel('time',font2)
    im+=1
    plt.xticks([0,48],['08:00','16:00'])
    
    plts=fig.add_subplot(6,4,im)
    draw(data[2],dfop[:,2],dfhc[:,2])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>20:
        plt.xlabel('time',font2)
    im+=1
    plt.xticks([0,48],['08:00','16:00'])
    
    plts=fig.add_subplot(6,4,im)
    draw(data[3],dfop[:,3],dfhc[:,3])
    if im<5:
        plt.title('Rain'+str(im),fontdict=font3)
    if im>20:
        plt.xlabel('time',font2)
    im+=1
    plt.xticks([0,48],['08:00','16:00'])
    
    line=line+1
  

plt.text(-210, 90, 'CSO and flooding volume (10$^{3}$ m$^{3}$)',rotation=90,fontdict=font3)

fig.savefig('5.5.2.png', bbox_inches='tight', dpi=500)
