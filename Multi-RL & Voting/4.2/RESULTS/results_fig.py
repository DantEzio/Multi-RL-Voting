
import pandas as pd
import matplotlib.pyplot as plt

testid='test'

dfa2c=pd.read_csv('./Final result/a2c_test_result/'+testid+' a2cflooding_vs_t.csv').values
dfddqn=pd.read_csv('./Final result/ddqn_test_result/'+testid+' ddqnflooding_vs_t.csv').values
dfdqn=pd.read_csv('./Final result/dqn_test_result/'+testid+' dqnflooding_vs_t.csv').values
dfppo1=pd.read_csv('./Final result/ppo1_test_result/'+testid+' ppo1flooding_vs_t.csv').values
dfppo2=pd.read_csv('./Final result/ppo2_test_result/'+testid+' ppo2flooding_vs_t.csv').values
dfvt=pd.read_csv('./Final result/voting_test_result/'+testid+' votingflooding_vs_t.csv').values
dfhc=pd.read_csv('./Final result/HC_test_result/'+testid+' hcflooding_vs_t.csv').values
dfop=pd.read_csv('./Final result/opt_test_result/'+testid+' optflooding_vs_t.csv').values

font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 10,}

font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 18,}

fig = plt.figure(figsize=(15,15))
for im in range(1,5):
    plts=fig.add_subplot(2,2,im)
    plt.plot(dfddqn[:,im-1],'r:',label='DDQN')
    plt.plot(dfdqn[:,im-1],'g:',label='DQN')
    plt.plot(dfppo1[:,im-1],'b:',label='PPO1')
    plt.plot(dfppo2[:,im-1],'c:',label='PPO2')
    plt.plot(dfa2c[:,im-1],'m:',label='A2C')
    plt.plot(dfvt[:,im-1],'y:',label='Voting')
    plt.plot(dfhc[:,im-1],'k.-',label='Water level system')
    plt.plot(dfop[:,im-1],'k',label='Optimization model')
    
    plt.xticks([0,48],['08:00','16:00'])
    plt.xlabel('time',font2)
    plt.ylabel('CSO volume (10$^{3}$ m$^{3}$)',font2)

    plt.legend(prop=font1)

fig.savefig(testid+'5.1.1.png', bbox_inches='tight', dpi=500)