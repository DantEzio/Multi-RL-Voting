import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fig():
    test='r2SC'
    rains=['0','1','2','3']
    node='_N1'
    for rain in rains:
        dfa2c=np.array(pd.read_excel('./excel results final/'+test+node+'_a2c.xlsx',sheet_name=rain))
        dfddqn=np.array(pd.read_excel('./excel results final/'+test+node+'_ddqn.xlsx',sheet_name=rain))
        dfdqn=np.array(pd.read_excel('./excel results final/'+test+node+'_dqn.xlsx',sheet_name=rain))
        dfppo1=np.array(pd.read_excel('./excel results final/'+test+node+'_ppo1.xlsx',sheet_name=rain))
        dfppo2=np.array(pd.read_excel('./excel results final/'+test+node+'_ppo2.xlsx',sheet_name=rain))
        dfvt=np.array(pd.read_excel('./excel results final/'+test+node+'_voting.xlsx',sheet_name=rain))
        #dfhc=np.array(pd.read_excel('./'+test+'_hc.xlsx'))
        
        print(dfa2c.shape,
              dfddqn.shape,
              dfdqn.shape,
              dfppo1.shape,
              dfppo2.shape,
              dfvt.shape)
              #dfhc.shape)
        
        
        for i in range(1):
            plt.figure()
            
            plt.plot(dfddqn[:,i],'r:',label='DDQN')
            plt.plot(dfdqn[:,i],'g:',label='DQN')
            plt.plot(dfppo1[:,i],'b:',label='PPO1')
            plt.plot(dfppo2[:,i],'c:',label='PPO2')
            plt.plot(dfa2c[:,i],'k:',label='A2C')
            plt.plot(dfvt[:,i],'b.-',label='voting system')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('water level')
            
def fig_pump():
    test='r1'
    rains=['0','1','2','3']
    
    for rain in rains:
        dfa2c=np.array(pd.read_excel('./'+test+'_a2c.xlsx',sheetname=rain))
        dfddqn=np.array(pd.read_excel('./'+test+'_ddqn.xlsx',sheetname=rain))
        dfdqn=np.array(pd.read_excel('./'+test+'_dqn.xlsx',sheetname=rain))
        dfppo1=np.array(pd.read_excel('./'+test+'_ppo1.xlsx',sheetname=rain))
        dfppo2=np.array(pd.read_excel('./'+test+'_ppo2.xlsx',sheetname=rain))
        dfvt=np.array(pd.read_excel('./'+test+'_voting.xlsx',sheetname=rain))
        dfhc=np.array(pd.read_excel('./hc.xlsx'))
        
        print(dfa2c.shape,
              dfddqn.shape,
              dfdqn.shape,
              dfppo1.shape,
              dfppo2.shape,
              dfvt.shape,
              dfhc.shape)
        
        
        for i in [0,3]:
            plt.figure()
            plt.plot(dfddqn[:,i],'r:',label='DDQN')
            plt.plot(dfddqn[:,i],'g:',label='DDQN pump')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('water level/pump signal')
            
def statistic():
    #time which the voting system achieve lower water level than others
    test='r1'
    rains=['0','1','2','3']
    
    for rain in rains:
        dfa2c=np.array(pd.read_excel('./excel results final/'+test+'_a2c.xlsx',sheetname=rain))
        dfddqn=np.array(pd.read_excel('./excel results final/'+test+'_ddqn.xlsx',sheetname=rain))
        dfdqn=np.array(pd.read_excel('./excel results final/'+test+'_dqn.xlsx',sheetname=rain))
        dfppo1=np.array(pd.read_excel('./excel results final/'+test+'_ppo1.xlsx',sheetname=rain))
        dfppo2=np.array(pd.read_excel('./excel results final/'+test+'_ppo2.xlsx',sheetname=rain))
        dfvt=np.array(pd.read_excel('./excel results final/'+test+'_voting.xlsx',sheetname=rain))
        dfhc=np.array(pd.read_excel('./excel results final/'+test+'hc.xlsx'))
        
        print(dfa2c.shape,
              dfddqn.shape,
              dfdqn.shape,
              dfppo1.shape,
              dfppo2.shape,
              dfvt.shape,
              dfhc.shape)
        forebay=['C','K','R']
        for i in range(3):
        
            data={'ddqn':dfddqn[:,i],'dqn':dfdqn[:,i],'ppo1':dfppo1[:,i],
                  'ppo2':dfppo2[:,i],'a2c':dfa2c[:,i],'voting':dfvt[:,i]}
            pddata=pd.DataFrame(data)
            pddata.plot.box(title='rain'+rain+' '+forebay[i])

        #pd.DataFrame(dfddqn).plot.box(title='DDQN')
        #pd.DataFrame(dfdqn[:,i]).plot.box(title='DQN')
        #pd.DataFrame(dfppo1[:,i]).plot.box(title='PPO1')
        '''
        plt.plot(dfppo2[:,i],'c:',label='PPO2')
        plt.plot(dfa2c[:,i],'k:',label='A2C')
        plt.plot(dfvt[:,i],'b.-',label='voting system')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('water level')
        '''

def statistic_SC():
    '''
    test='r2SC'
    rains=['0','1','2','3']
    node='_N1'
    
    if node=='_N1':
        sl=2.21*0.7
    else:
        sl=1.957*0.7
    '''
    nums=['0','1','2','3','4','5','6','7']
    nms=['dqn_test_result','ddqn_test_result','ppo1_test_result',
         'ppo2_test_result','a2c_test_result','voting_test_result']
    
    ps=['N1','N2']#['1','2','3','4']
    
    #N1点
    p=ps[0]
    
    st1,st2,st3=[],[],[]
    for nm in nms:
        line1r1,line2r1,line3r1=[],[],[]
        for num in nums:
            dftem1=np.array(pd.read_excel('./excel results final/RESULTS_'+p+'_'+nm+'.xlsx',sheet_name=num))

            line1r1.append(np.around(np.mean(dftem1),decimals=4))
            line2r1.append(np.around(np.var(dftem1),decimals=4))
            line3r1.append(np.around(np.median(dftem1),decimals=4))
        
        line1=line1r1
        line2=line2r1
        line3=line3r1
        
        st1.append(line1)
        st2.append(line2)
        st3.append(line3)

    meandf_N1=pd.DataFrame(st1)
    vardf_N1=pd.DataFrame(st2)
    meddf_N1=pd.DataFrame(st3)

    meandf_N1.to_excel('./water level_N'+p+'_mean_.xlsx', sheet_name = 'mean')
    vardf_N1.to_excel('./water level_N'+p+'_var_.xlsx', sheet_name = 'var')
    meddf_N1.to_excel('./water level_N'+p+'_median_.xlsx', sheet_name = 'median')
    
                    
if __name__=='__main__':
    #fig()
    statistic_SC()     
