import pandas as pd

actions=[]
for i1 in [0,1]:
    for i2 in [0,1]:
        for i3 in [0,1]:
            for i4 in [0,1]:
                for i5 in [0,1]:
                    for i6 in [0,1]:
                        for i7 in [0,1]:
                            for i8 in [0,1]:
                                actions.append([i1,i2,i3,i4,i5,i6,i7,i8])

d=pd.DataFrame(actions)
writer = pd.ExcelWriter('./action_table_of_DQN.xlsx')
d.to_excel(writer)
writer.save()

actions=pd.read_excel('./action_table_of_DQN.xlsx').values[:,:6]
print(actions.shape)