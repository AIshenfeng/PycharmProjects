from multiprocessing.managers import BaseManager
import time, queue, random

task_queue = queue.Queue()
result_queue = queue.Queue()


class QueueManager(BaseManager):
    pass


QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)

manager = QueueManager(('', 5000), authkey=b'abc')
manager.start()

task = manager.get_task_queue()
result = manager.get_result_queue()

for i in range(10):
    n = random.randint(0, 100)
    task.put(n)

for j in range(10):
    r = result.get(timeout=10)
    print(r)



