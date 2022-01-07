import os
import struct
import numpy as np
import pandas as pd

from pyswmm import Simulation


def read_out(filename):
    '''
    t:which time point you want to return 
    '''
    RECORDSIZE=4
    version=0
    NflowUnits=0
    Nsubcatch=0
    Nnodes=0
    Nlinks=0
    Npolluts=0
    
    
    magic1=0
    magic2=0
    magic3=0
    err=0
    startPos=0
    nPeriods=0
    errCode=0
    IDpos=0
    propertyPos=0
    
    pollutantUnit=''
    
    sub_name=[]
    node_name=[]
    link_name=[]
    poll_name=[]
    reportInterval=[]
    subcatchResultValueList=[]
    nodeResultValueList=[]
    linkResultValueList=[]
    systemResultValueList=[]
    data={}
    
    #各个要素id
    
    br=open(filename,'rb')
    #判断文件是否正常打开
    if(br==None or os.path.getsize(filename)):
        err=1
    
    #读取末尾的位置属性
    br.seek(os.path.getsize(filename)-RECORDSIZE*6)
    IDpos=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    propertyPos=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    startPos=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    nPeriods=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    errCode=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    magic2=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    
    #print(IDpos,propertyPos,startPos,nPeriods,errCode,magic2)
    
    #读取开头的magic变量
    br.seek(0)
    magic1=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
    
    if(magic1!=magic2 or errCode!=0 or nPeriods==0):
        err=1
    else:
        err=0
        
    if(err==1):
        br.close()
        return sub_data,node_data,link_data
    else:
            
        #读取版本号，单位，汇水区个数，节点个数，管道个数，污染物个数
        version=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        NflowUnits=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Nsubcatch=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Nnodes=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Nlinks=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        Npolluts=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        
        #print(version,NflowUnits,Nsubcatch,Nnodes,Nlinks,Npolluts)
        
        #读取各个id列表
        br.seek(IDpos)
        
        
        for i in range(Nsubcatch):
            numSubIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            subcatchByte=br.read(numSubIdNames)
            sub_name.append(subcatchByte.decode(encoding = "utf-8"))
        
        for i in range(Nnodes):
            numNodeIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            nodeByte=br.read(numNodeIdNames)
            node_name.append(nodeByte.decode(encoding = "utf-8"))
        
        for i in range(Nlinks):
            numlinkIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            linkByte=br.read(numlinkIdNames)
            link_name.append(linkByte.decode(encoding = "utf-8"))
        
        for i in range(Npolluts):
            numpollutsIdNames=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
            pollutsByte=br.read(numpollutsIdNames)
            poll_name.append(pollutsByte.decode(encoding = "utf-8"))
        
        #读取污染物单位
        unit=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        #print(unit)
        if unit==0:
            pollutantUnit='mg/L'
        if unit==1:
            pollutantUnit='ug/L'
        if unit==2:
            pollutantUnit='counts/L'
         
        #读取各个属性个数
        br.seek(propertyPos)
        numSubcatProperty=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        offsetTemp1=numSubcatProperty*Nsubcatch
        br.seek((offsetTemp1+1)*4,1)
        numNodeProperty=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        offsetTemp2=numNodeProperty*Nnodes
        br.seek((offsetTemp2+3)*4,1)
        numLinkProperty=int.from_bytes(br.read(RECORDSIZE),byteorder='little')
        
        #print(numSubcatProperty,numNodeProperty,numLinkProperty)
        
        #读取各个属性
        subcatchProNameList=[]
        subcatchProValueList=[]
        nodeProNameList=[]
        nodeProValueList=[]
        linkProNameList=[]
        linkProValueList=[]
        
        br.seek(propertyPos+4)
        subcatchProNameList.append(int.from_bytes(br.read(RECORDSIZE),byteorder='little'))
        for i in range(Nsubcatch):
            subcatchProValueList.append(struct.unpack('f',br.read(RECORDSIZE)))
            #txtSubcatchPro.Text+=subcatchProValueList[i].To
        
        br.read(RECORDSIZE)
        for k in range(3):
            nodeProNameList.append(int.from_bytes(br.read(RECORDSIZE),byteorder='little'))
        for i in range(Nnodes*3):
            nodeProValueList.append(struct.unpack('f',br.read(RECORDSIZE)))
            
        #print(nodeProValueList)
            
        br.read(RECORDSIZE)
        for k in range(5):
            linkProNameList.append(int.from_bytes(br.read(RECORDSIZE),byteorder='little'))
        for i in range(Nlinks*5):
            linkProValueList.append(struct.unpack('f',br.read(RECORDSIZE)))
        
        #print(nodeProValueList)
        
        '''
        computing result
        '''
        #读取计算结果
        br.seek(startPos)   
        for i in range(nPeriods):
            dt=struct.unpack('f',br.read(RECORDSIZE))
            reportInterval.append(dt)
            br.read(RECORDSIZE)
            tem=[]
            for su in range(Nsubcatch):
                tem1=[]
                for su1 in range(8+Npolluts):
                    tem1.append(struct.unpack('f',br.read(RECORDSIZE)))
                tem.append(tem1)
            subcatchResultValueList.append(tem)
        
            tem=[]
            for no in range(Nnodes):
                tem1=[]
                for no1 in range(6+Npolluts):
                    tem1.append(struct.unpack('f',br.read(RECORDSIZE)))
                tem.append(tem1)
            nodeResultValueList.append(tem)
            
            tem=[]
            for li in range(Nlinks):
                tem1=[]
                for li1 in range(5+Npolluts):
                    tem1.append(struct.unpack('f',br.read(RECORDSIZE)))
                tem.append(tem1)
            linkResultValueList.append(tem)
        
            tem=[]
            for sy in range(15):
                tem.append(struct.unpack('f',br.read(RECORDSIZE)))
            systemResultValueList.append(tem)
        
        br.close()   
        ''' 
        k=0
        for item in sub_name:
            sub_data[item]=subcatchResultValueList[-1][k]
            k+=1
        k=0
        for item in link_name:
            data[item]=linkResultValueList[t][k][0]
            k+=1
        k=0
        for item in node_name:
            data[item]=[nodeResultValueList[t][k][6],nodeResultValueList[t][k][0]]
            k+=1
        '''
        
        return link_name,node_name,linkResultValueList,nodeResultValueList,node_name

'''
测试方法
'''
def get_depth(o_data,node_name,t):
    '''
    t时刻所有节点的水位
    '''
    k=0
    depth={}
    for item in node_name:
        depth[item]=o_data[t][k][0][0]
        k+=1
    return depth

def get_head(o_data,node_name,t):
    '''
    t时刻所有节点的Head
    '''
    k=0
    head={}
    for item in node_name:
        head[item]=o_data[t][k][1][0]
        k+=1
    return head


def get_volume(o_data,node_name,t):
    '''
    t时刻所有节点的volume
    '''
    k=0
    volume={}
    for item in node_name:
        volume[item]=o_data[t][k][0][0]
        k+=1
    return volume


def get_lateral_inflow(o_data,node_name,t):
    '''
    t时刻所有节点流入量
    '''
    k=0
    inflow={}
    for item in node_name:
        inflow[item]=o_data[t][k][3][0]
        k+=1
    return inflow

def get_total_inflow(o_data,node_name,t):
    '''
    到t时刻所有节点的总流量
    '''
    k=0
    t_inflow={}
    for item in node_name:
        t_inflow[item]=o_data[t][k][4][0]
        k+=1
    return t_inflow


def get_flood(o_data,node_name,t):
    '''
    t时刻所有节点的flooding
    '''
    k=0
    flood={}
    for item in node_name:
        flood[item]=o_data[t][k][5][0]
        k+=1
    return flood

def get_cod(o_data,node_name,t):
    '''
    t时刻所有节点COD
    '''
    k=0
    COD={}
    for item in node_name:
        COD[item]=o_data[t][k][6][0]
        k+=1
    return COD



'''
功能解析方法
'''
def node_depth(ndata,index,NN):
    '''
    t时刻out文件中的前池水位
    '''
    T=len(ndata)
    N=len(ndata[0])
    M=len(ndata[0][0])
    print(T,N,M)
    #pool_list=['CC-storage','JK-storage','XR-storage']
    #pool_list=['WS02006251']
    #pool_ind=[-2,-1,-3]
    pool_list=[NN]
    pool_ind=[index]
    pool_d={}
    for it in range(len(pool_list)):
        tem=[]
        for t in range(T):
            tem.append(ndata[t][pool_ind[it]][0][0])
        pool_d[pool_list[it]]=tem
    return pool_d

def pump_flow(ldata):
    '''
    t时刻out文件中的pump curve
    '''
    T=len(ldata)
    N=len(ldata[0])
    M=len(ldata[0][0])
    print(T,N,M)
    
    pump_list=['CC-Pump-1','CC-Pump-2','JK-Pump-1','JK-Pump-2','XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']
    pump_ind=[-4,-3,-2,-1,-8,-6,-7,-5]
    pump_f={}
    for it in range(len(pump_list)):
        tem=[]
        for t in range(T):
            a=0
            if ldata[t][pump_ind[it]][0][0]>0:
                a=1          
            tem.append(a)
        pump_f[pump_list[it]]=tem

    return pump_f

def save_txt():
    nums=['3','4','6']
    
    for num in nums:
        print(num)
        
        filename='./mpc/'+num+'.out'
        lname,nname,ldata,ndata,name=read_out(filename)
        
        #将data写入txt
        n_depth=node_depth(ndata)
        T=len(ndata)
        #读取前池水位
        text1=text2=text3=''
        for t in range(T):
            text1+=str(n_depth['CC-storage'][t])+'\n'
            text2+=str(n_depth['JK-storage'][t])+'\n'
            text3+=str(n_depth['XR-storage'][t])+'\n'
    
        output = open('./'+num+'/pool/CC-storage.txt','wt')
        output.write(text1)
        output.close()
        output = open('./'+num+'/pool/JK-storage.txt','wt')
        output.write(text2)
        output.close()
        output = open('./'+num+'/pool/XR-storage.txt','wt')
        output.write(text3)
        output.close()
        
        #将data写入txt
        p_flow=pump_flow(ldata)
        T=len(ldata)
        #读取泵曲线
        text1=text2=text3=text4=text5=text6=text7=text8=''
        for t in range(T):
            text1+=str(p_flow['CC-Pump-1'][t])+'\n'
            text2+=str(p_flow['CC-Pump-2'][t])+'\n'
            text3+=str(p_flow['JK-Pump-1'][t])+'\n'
            text4+=str(p_flow['JK-Pump-2'][t])+'\n'
            text5+=str(p_flow['XR-Pump-1'][t])+'\n'
            text6+=str(p_flow['XR-Pump-2'][t])+'\n'
            text7+=str(p_flow['XR-Pump-3'][t])+'\n'
            text8+=str(p_flow['XR-Pump-4'][t])+'\n'
    
        output = open('./'+num+'/pump/CC-Pump-1.txt','wt')
        output.write(text1)
        output.close()
        output = open('./'+num+'/pump/CC-Pump-2.txt','wt')
        output.write(text2)
        output.close()
        output = open('./'+num+'/pump/JK-Pump-1.txt','wt')
        output.write(text3)
        output.close()
        output = open('./'+num+'/pump/JK-Pump-2.txt','wt')
        output.write(text4)
        output.close()
        output = open('./'+num+'/pump/XR-Pump-1.txt','wt')
        output.write(text5)
        output.close()
        output = open('./'+num+'/pump/XR-Pump-2.txt','wt')
        output.write(text6)
        output.close()
        output = open('./'+num+'/pump/XR-Pump-3.txt','wt')
        output.write(text7)
        output.close()
        output = open('./'+num+'/pump/XR-Pump-4.txt','wt')
        output.write(text8)
        output.close()


def save_xls():
    nums=['0','1','2','3']
    writer=pd.ExcelWriter('./r1_a2c.xlsx')
    for num in nums:
        print(num)
        
        filename='./results1/a2c_test_result/'+num+'.out'
        lname,nname,ldata,ndata,name=read_out(filename)
        
        #将data写入txt
        n_depth=node_depth(ndata)
        p_flow=pump_flow(ldata)
        T=len(ndata)
        
        data=[]
        for t in range(T):
            tem=[]
            tem.append(n_depth['CC-storage'][t])
            tem.append(n_depth['JK-storage'][t])
            tem.append(n_depth['XR-storage'][t])
    
            tem.append(p_flow['CC-Pump-1'][t]+p_flow['CC-Pump-2'][t])
            tem.append(p_flow['JK-Pump-1'][t]+p_flow['JK-Pump-2'][t])
            tem.append(p_flow['XR-Pump-1'][t]+p_flow['XR-Pump-2'][t]+p_flow['XR-Pump-3'][t]+p_flow['XR-Pump-4'][t])
            data.append(tem)
        
        #output = open('./RLCdata'+num+'.txt','wt')
        #output.write(data)
        #output.close()
        
        D=np.array(data)
        xls_pd=pd.DataFrame(D)
        xls_pd.to_excel(writer,sheet_name=str(num),header=False,index=False)
    writer.save()
    
def savehc_xls():
    nums=['0','1','2','3']
    writer=pd.ExcelWriter('./hc.xlsx')
    for num in nums:
        print(num)
        
        filename='./result_hc/HC'+num+'.out'
        lname,nname,ldata,ndata,name=read_out(filename)
        
        #将data写入txt
        n_depth=node_depth(ndata)
        p_flow=pump_flow(ldata)
        T=len(ndata)
        
        data=[]
        for t in range(T):
            tem=[]
            tem.append(n_depth['CC-storage'][t])
            tem.append(n_depth['JK-storage'][t])
            tem.append(n_depth['XR-storage'][t])
    
            tem.append(p_flow['CC-Pump-1'][t]+p_flow['CC-Pump-2'][t])
            tem.append(p_flow['JK-Pump-1'][t]+p_flow['JK-Pump-2'][t])
            tem.append(p_flow['XR-Pump-1'][t]+p_flow['XR-Pump-2'][t]+p_flow['XR-Pump-3'][t]+p_flow['XR-Pump-4'][t])
            data.append(tem)
        
        #output = open('./RLCdata'+num+'.txt','wt')
        #output.write(data)
        #output.close()
        
        D=np.array(data)
        xls_pd=pd.DataFrame(D)
        xls_pd.to_excel(writer,sheet_name=str(num),header=False,index=False)
    writer.save()


###############################################################################

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass    
    
    
def saveSC_xls():
    nums=['0','1','2','3','4','5','6','7']
    nms=['dqn_test_result','ddqn_test_result','ppo1_test_result',
         'ppo2_test_result','a2c_test_result','voting_test_result']
    #Ns=['CC-storage','JK-storage']
    Ns=['WS02006235','WS02006252']#'YS02001757']#
    tests=['RESULTS']
    
    ps=['1','2']
    
    for test in tests:
        for nm in nms:
            print(test,nm)
            for p in ps:
                writer=pd.ExcelWriter('./excel results final/'+test+'_N'+p+'_'+nm+'.xlsx')
                for num in nums:
                    filename='./'+test+'/Final result/'+nm+'/'+num+'.inp'
                    simulation(filename)
                    
                    filename='./'+test+'/Final result/'+nm+'/'+num+'.out'
                    lname,nname,ldata,ndata,name=read_out(filename)
                    N=Ns[int(p)-1]
                    n_depth=node_depth(ndata,name.index(N),N)
                    #p_flow=pump_flow(ldata)
                    T=len(ndata)
                    
                    data=[]
                    for t in range(T):
                        tem=[]
                        #tem.append(n_depth['WS02006251'][t])
                        tem.append(n_depth[N][t])
                
                        data.append(tem)
                    
                    #output = open('./RLCdata'+num+'.txt','wt')
                    #output.write(data)
                    #output.close()
                    
                    D=np.array(data)
                    xls_pd=pd.DataFrame(D)
                    xls_pd.to_excel(writer,sheet_name=str(num),header=False,index=False)
                writer.save()
        

if __name__=='__main__':
    #save_txt()
    saveSC_xls()

    '''
    
    print(name)
    print(".................................")
    print(get_cod(data,name,t))
    print(".................................")
    print(get_depth(data,name,t))
    print(".................................")
    print(get_flood(data,name,t))
    print(".................................")
    print(get_total_inflow(data,name,t))
    print(".................................")
    print(get_lateral_inflow(data,name,t))
    print(".................................")
    print(get_volume(data,name,t))
    print(".................................")
    print(get_head(data,name,t))
    
    pool_list=['CC-storage']

    print(depth(filename,pool_list,t))
    '''
    