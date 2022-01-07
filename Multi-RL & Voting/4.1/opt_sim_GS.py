import numpy as np
import get_rpt
import set_datetime
import xlrd

import get_output#从out文件读取水动力初值
import set_pump#生成下一时段的inp
import change_rain#随机生成降雨序列

import random

import datetime
import pandas as pd

from pyswmm import Simulation
from swmm_api.input_file import read_inp_file, SwmmInput, section_labels as sections


def discount_reward(r):
    gamma=0.99
    discounted_r=np.zeros_like(r)
    running_add=0
    for t in reversed(range(r.size)):
        running_add=running_add*gamma+r[t]
        discounted_r[t]=running_add
    return discounted_r

def simulation(filename):
    with Simulation(filename) as sim:
        #stand_reward=0
        for step in sim:
            pass    

def copy_result(outfile,infile):
    output = open(outfile, 'wt')
    with open(infile, 'rt') as data:
        for line in data:
            output.write(line)
    output.close()


def read_data(st):
    tr_data=[]
    data=xlrd.open_workbook(st)
    table=data.sheets()[0]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(nrows):
        tem=[]
        for j in range(ncols):
            tem.append(table.cell(i,j).value)     
        tr_data.append(tem)
    t_data=np.array(tr_data)
    t_data.reshape(nrows,ncols)
    print(t_data.shape)
    return t_data


#########################################################################################
def initPopulation(lifeCount,rainid):
    """初始化种群"""
    lives = []
    inp = SwmmInput.read_file('./dqn_test_result/'+str(0)+'.inp')
    T=len(inp[sections.TIMESERIES]['pump_0'].data)
    Pumps=['pump_0','pump_1','pump_2','pump_3','pump_4','pump_5','pump_6','pump_7']
    
    for i in range(lifeCount):
        gene=[0 for _ in range(len(Pumps))]
        for j in range(T):
            for p in Pumps:
                gene.append(inp[sections.TIMESERIES][p].data[j][1])
            
        for _ in range(2*len(Pumps)):
            gene.append(0)
        lives.append(gene)
    return lives


def cross(parent1, parent2):
    """交叉"""
    geneLength=len(parent1)
    index1 = random.randint(0, geneLength - 1)
    index2 = random.randint(index1, geneLength - 1)
    tempGene = parent2[index1:index2]                      #交叉的基因片段
    newGene = parent1
    newGene[index1:index2]=tempGene
    return newGene


def  mutation(gene):
    """突变"""
    geneLength=len(gene)
    index1 = random.randint(0, geneLength - 1)
    index2 = random.randint(0, geneLength - 1)
    t=gene[index1]
    gene[index1]=gene[index2]
    gene[index2]=t
    return gene


def getOne(lives,scores,bounds):
    """选择一个个体"""
    r = random.uniform(0, bounds)
    for i in range(len(lives)):
        r -= scores[i]
        if r <= 0:
            return lives[i]

    raise Exception("选择错误", bounds)


def newChild(crossRate,mutationRate,lives,scores,bounds):
    """产生新后的"""
    parent1 = getOne(lives,scores,bounds)
    rate = random.random()

    #按概率交叉
    if rate < crossRate:
        #交叉
        parent2 = getOne(lives,scores,bounds)
        gene = cross(parent1, parent2)
    else:
        gene = parent1

    #按概率突变
    rate = random.random()
    if rate < mutationRate:
        gene = mutation(gene)

    return gene


#修改降雨数据，进行多场降雨模拟计算
def opt_GA_sim(startfile,simfile,crossRate,mutationRate,lifeCount,stepNum,date_time,pumps,raindata_ind,raindata):
    
    
    if raindata=='test':
        rainData=np.loadtxt('./sim/testRainFile.txt',delimiter=',')#读取测试降雨数据
    else:
        raindata_ind-=4
        rainData=np.loadtxt('./sim/real_rain_data.txt',delimiter=' ')*20#读取真实降雨数据
    rainnum,m=rainData.shape
    rainData=np.hstack((rainData,np.zeros((rainnum,m))))    
    
    change_rain.change_rain(rainData[raindata_ind],startfile+'.inp')
    change_rain.copy_result(simfile+'.inp',startfile+'.inp')#将修改了rain数据的infile inp文件进行复制    
    action_seq=[]

    #初始化
    lives=initPopulation(lifeCount,raindata_ind)
    print('lives shape: ',len(lives),len(lives[0]))
    scores=[]
    bounds=0
    generation=0
    
    #init, first step
    for gene in lives:
        tem=np.array(gene)
        action_seq=list(tem.reshape(len(date_time),len(pumps)))#25*8的数组
        change_rain.copy_result(simfile+'.inp',startfile+'.inp')#将startfile复制为infile
        set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
        simulation(simfile+'.inp')
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
        if total_in==0:
            score=0
        else:
            score=(total_in-flooding)/(total_in)
        scores.append(score)
        
        bounds+=score
    best=lives[scores.index(max(scores))]
    
    log_score=[]
    for i in range(stepNum):
        if np.mod(i,10)==0:
            print('steps: ',i)
        newLives = []
        newLives.append(best)#把最好的个体加入下一代
        while len(newLives) < lifeCount:
            newLives.append(newChild(crossRate,mutationRate,lives,scores,bounds))
        lives = newLives
        generation += 1
        
        scores=[]
        bounds=0
        #print('step'+str(i))
        for gene in lives:
            #print(action_seq)
            tem=np.array(gene)
            action_seq=list(tem.reshape(len(date_time),len(pumps)))
            
            change_rain.copy_result(simfile+'.inp',startfile+'.inp')
            set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
            simulation(simfile+'.inp')
            #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
            if total_in==0:
                score=0
            else:
                score=(total_in-flooding)/(total_in)
            scores.append(score)
            
            bounds+=score
        best=lives[scores.index(max(scores))]
        max_scors=max(scores)
        log_score.append(max_scors)
        #print(i,'  ',end-begin)
        
    #最佳策略的模拟结果
    tem=np.array(best)
    action_seq=tem.reshape(len(date_time),len(pumps))     
    change_rain.copy_result(simfile+'.inp',startfile+'.inp')
    set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')
    flooding_logs=[]
    sdate=edate='08/28/2015'
    stime=date_time[0]
    for t in date_time[1:]:
        tem_etime=t
        set_datetime.set_date(sdate,edate,stime,tem_etime,simfile+'.inp')
        simulation(simfile+'.inp')
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
        flooding_logs.append(flooding)
    #print('search done, time: ',end-begin)
    
    #保存训练的inp与rpt文件
    copy_result('./opt_test_result/opt_'+str(raindata_ind)+'.inp',simfile+'.inp')
    copy_result('./opt_test_result/opt_'+str(raindata_ind)+'.rpt',simfile+'.rpt')
    return action_seq,flooding_logs

if __name__=='__main__':
    startfile='./sim/orf'
    simfile='./sim/opt_using'
    crossRate,mutationRate,lifeCount,stepNum=0.01,0.03,100,1
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00']
    pumps=['CC-Pump-1','CC-Pump-2','JK-Pump-1','JK-Pump-2','XR-Pump-1','XR-Pump-2','XR-Pump-3','XR-Pump-4']
    
    flooding_logs=[]
    for raindata_ind in range(8):
        print('rain num: ',raindata_ind)
        if raindata_ind<4:
            rainid='test'
        else:
            rainid='real'
        aq,fls=opt_GA_sim(startfile,simfile,crossRate,mutationRate,lifeCount,stepNum,date_time,pumps,raindata_ind,rainid)
        np.savetxt('./opt_test_result/opt_'+str(raindata_ind)+'.txt', aq, fmt="%d")
        flooding_logs.append(fls)
    
    
    #   #保存所有降雨的flooding过程线
    df = pd.DataFrame(np.array(flooding_logs).T)
    df.to_excel('./opt_test_result/opt_flooding_vs_t.xlsx', index=False, encoding='utf-8')
        

    
    