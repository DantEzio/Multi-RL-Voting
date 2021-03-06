import PPO
import env_SWMM
import tensorflow as tf


if __name__=='__main__':
    
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
    
    AFI=False
    num_rain=4
    test_num=4
    step=10
    
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    g_3 = tf.Graph()
    with g_3.as_default():
        model3 = PPO.PPO(env,step, 240, num_rain, 'ppo1', raindata='test')
        model3.load_model()
        model3.train(True)
        model3.test(test_num)
    
    g_31 = tf.Graph()
    with g_31.as_default():
        model3 = PPO.PPO(env,step, 240, num_rain, 'ppo1',raindata='real')
        model3.load_model()
        model3.test(test_num)
    
    env=env_SWMM.env_SWMM(date_time, date_t,AFI)
    g_4 = tf.Graph()
    with g_4.as_default():
        model4 = PPO.PPO(env,step, 240, num_rain, 'ppo2',raindata='test')
        model4.load_model()
        model4.train(True)
        model4.test(test_num)
    
    g_41 = tf.Graph()
    with g_41.as_default():    
        model4 = PPO.PPO(env,step, 240, num_rain, 'ppo2', raindata='real')
        model4.load_model()
        model4.test(test_num)

    