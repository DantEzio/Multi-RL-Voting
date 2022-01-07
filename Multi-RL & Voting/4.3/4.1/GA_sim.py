import numpy as np
import get_rpt
import set_datetime
import xlrd

import get_output#从out文件读取水动力初值
import set_pump#生成下一时段的inp
import change_rain#随机生成降雨序列

import random
import datetime
from pyswmm import Simulation

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
def initPopulation(lifeCount,geneLength):
    """初始化种群"""
    lives = []
    for i in range(lifeCount):
        gene=[]
        for j in range(geneLength):
            #gene.append(random.randint(0,1))
            gene.append(1)
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
    #突变次数加1
    #mutationCount += 1
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

def GA_sim(startfile,simfile,crossRate,mutationRate,lifeCount,date_time,pumps,stepNum):       
    iten=0
    iten+=1
    change_rain.copy_result(simfile+'.inp',startfile+'.inp')#将修改了rain数据的infile inp文件进行复制    
    action_seq=[]
    t_reward=[]
    begin = datetime.datetime.now()
    #初始化
    lives=initPopulation(lifeCount,len(date_time)*len(pumps))
    scores=[]
    bounds=0
    generation=0
    for gene in lives:      
        tem=np.array(gene)
        action_seq=list(tem.reshape(len(date_time),len(pumps)))
        #print(action_seq)
        change_rain.copy_result(simfile+'.inp',startfile+'.inp')#将startfile复制为infile
        set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
        simulation(simfile+'.inp')
        #change_rain.copy_result('check'+str(i)+'.inp',infile+'.inp')
        #获取rpt内信息，产生新的action
        total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
        scores.append(1/(1+flooding))
        score=1/(1+flooding)
        bounds+=score
    best=lives[scores.index(max(scores))]        
    begin = datetime.datetime.now()
    
    for i in range(stepNum):        
        #评估，计算每一个个体的适配值
        newLives = []
        newLives.append(best)#把最好的个体加入下一代
        while len(newLives) < lifeCount:
            newLives.append(newChild(crossRate,mutationRate,lives,scores,bounds))
        lives = newLives
        generation += 1
        scores=[]
        bounds=0
        for gene in lives:
            tem=np.array(gene)
            action_seq=list(tem.reshape(len(date_time),len(pumps)))
            change_rain.copy_result(simfile+'.inp',startfile+'.inp')
            set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
            simulation(simfile+'.inp')
            total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
            score=1/(1+flooding)
            scores.append(score)            
            bounds+=score
        best=lives[scores.index(max(scores))]
        max_scors=max(scores)
        end = datetime.datetime.now()
        
    #最佳策略的模拟结果
    tem=np.array(best)
    action_seq=tem.reshape(len(date_time),len(pumps))
    change_rain.copy_result(simfile+'.inp',startfile+'.inp')
    set_pump.set_pump(action_seq,date_time[0:len(date_time)-1],pumps,simfile+'.inp')    
    simulation(simfile+'.inp')
    total_in,flooding,store,outflow,upflow,downflow=get_rpt.get_rpt(simfile+'.rpt')
    score=1/(1+flooding)
    
    end = datetime.datetime.now()        
    copy_result('./sim/GA/GA_'+str(iten)+'.inp',simfile+'.inp')
    copy_result('./sim/GA/GA_'+str(iten)+'.rpt',simfile+'.rpt')
    return action_seq
   
    
    
    