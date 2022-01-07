
import pandas as pd
import numpy as np

testid='real'#'test'

dfa2c=pd.read_csv('./Final result/a2c_test_result/'+testid+' a2cflooding_vs_t.csv').values
dfddqn=pd.read_csv('./Final result/ddqn_test_result/'+testid+' ddqnflooding_vs_t.csv').values
dfdqn=pd.read_csv('./Final result/dqn_test_result/'+testid+' dqnflooding_vs_t.csv').values
dfppo1=pd.read_csv('./Final result/ppo1_test_result/'+testid+' ppo1flooding_vs_t.csv').values
dfppo2=pd.read_csv('./Final result/ppo2_test_result/'+testid+' ppo2flooding_vs_t.csv').values
dfvt=pd.read_csv('./Final result/voting_test_result/'+testid+' votingflooding_vs_t.csv').values
dfhc=pd.read_csv('./Final result/HC_test_result/'+testid+' hcflooding_vs_t.csv').values
dfop=pd.read_csv('./Final result/opt_test_result/'+testid+' optflooding_vs_t.csv').values


data=np.concatenate((dfdqn[-1,:],
                     dfddqn[-1,:],
                     dfppo1[-1,:],
                     dfppo2[-1,:],
                     dfa2c[-1,:],
                     dfvt[-1,:],
                     dfhc[-1,:],
                     dfop[-1,:]),axis=0).reshape((8,4))
print(data.shape)
pd.DataFrame(data).to_csv(testid+'_results_table.csv')
