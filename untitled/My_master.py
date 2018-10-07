import time,random,queue
from multiprocessing.managers import BaseManager

#发送任务队列
task_queue=queue.Queue()
#接受结果队列
result_queue=queue.Queue()

#继承BaseManager
class Queuemanager(BaseManager):
    pass

#QueueManager要管理多个queue,给他们起一个名字
Queuemanager.register('get_task_queue',callable=lambda :task_queue)
Queuemanager.register('get_result_queue',callable=lambda :result_queue)

#manager实例化
manager=Queuemanager(address=('',5000),authkey=b'abc')
manager.start()

#通过网络获得task_queue而不能直接对task_queue操作
task=manager.get_task_queue()
result=manager.get_result_queue()

#添加任务
for i in range(10):
    task.put(random.randint(1,100))

#接收结果
for j in range(10):
    r=result.get(timeout=10)
    print(r)




