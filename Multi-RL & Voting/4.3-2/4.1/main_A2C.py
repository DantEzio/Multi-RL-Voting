import A2C
import env_SWMM
import tensorflow as tf


if __name__=='__main__':
    '''
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50','12:00']
    date_t=[0,10,20,30,40,50,\
            60,70,80,90,100,110,\
            120,130,140,150,160,170,\
            180,190,200,210,220,230,240]
    '''
    
    date_time=['08:00','08:10','08:20','08:30','08:40','08:50',\
               '09:00','09:10','09:20','09:30','09:40','09:50',\
               '10:00','10:10','10:20','10:30','10:40','10:50',\
               '11:00','11:10','11:20','11:30','11:40','11:50',\
               '12:00','12:10','12:20','12:30','12:40','12:50',\
               '13:00','13:10','13:20','13:30','13:40','13:50',\
               '14:00','14:10','14:20','14:30','14:40','14:50',\
               '15:00','15:10','15:20','15:30','15:40','15:50',\
               '16:00']
    
    date_t=[]
    for i in range(len(date_time)):
        date_t.append(int(i*10))
    
    
    '''
    print(observation)
    for t in range(len(date_t)-3):
        r=env.step([0.3,0.7])
        print(env.iten)
        print(r)
    '''
    
    
    test_num=4
    MAX_EPISODE = 20
    MAX_EP_STEPS = 240  # maximum time step in one episode
    num_rain=4
    AFI=False
    
    env=env_SWMM.env_SWMM(date_time, date_t, AFI)
    g_1 = tf.Graph()
    with g_1.as_default():
        model1=A2C.A2C(MAX_EPISODE,MAX_EP_STEPS,num_rain,env,'test')
        #history=model1.train(True)
        model1.load_model()
        for rid in range(20):
            r=model1.test(test_num,rid)
    
    
    