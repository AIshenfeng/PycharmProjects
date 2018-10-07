from multiprocessing.managers import BaseManager
import time

#创建管理
class QueueManager(BaseManager):
    pass

#要从网络上获取这两个队列，注册所需要的队列的名称
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

#实例化并连接网络
manager=QueueManager(address=('127.0.0.1',5000),authkey=b'abc')
manager.connect()

#获得队列
task=manager.get_task_queue()
result=manager.get_result_queue()


for i in range(10):
    try:
        #处理任务
        n=task.get(timeout=1)
        print('runing %d * %d ～～～' %(n,n))
        time.sleep(1)
        r='%d * %d =%d' %(n,n,n*n)
        #把处理结果压入结果队列
        result.put(r)
    except:
        print("task_queue is enpty")


