from multiprocessing.managers import BaseManager
import time

class QueueManager(BaseManager):
    pass
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

manager=QueueManager(('127.0.0.1',5000),authkey=b'abc')

manager.connect()

task_queue=manager.get_task_queue()
result_queue=manager.get_result_queue()

for i in range(10):
    try:
        n=task_queue.get(timeout=1)
        print('Running %d * %d ----'%(n,n))
        r='%d * %d = %d '%(n,n,n*n)
        time.sleep(1)
        result_queue.put(r)
    except:
        print("task_queue is empty!")

print('worker is over!')