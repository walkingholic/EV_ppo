import multiprocessing as mp
from multiprocessing import Lock, Queue, current_process
import time, os

def work(id, n, lock):
    for i in range(n):

        lock.acquire()
        print('[', id, i, ']')
        lock.release()
        time.sleep(0.5)

def creator(data, q):

    print('Create data and put it on the Q')
    for item in data:
        q.put(item)
        print('put item ', item)
        time.sleep(1)

def consumer(q):
    while True:
        data = q.get()
        processed = data*10
        print('{} data found to be processed {} -> {}'.format(current_process().name, data, processed))

        # if q.empty():
        #     break

def adder(id, n, lock):

    for i in range(5000):
        val = n.value
        with n.get_lock():
            n.value += 1
        #     print(id, val, n.value)
        # time.sleep(0.5)




class Worker(mp.Process):
    def __init__(self, i, num, until):
        super().__init__()
        self.id, self.num, self.until = i, num, until

    def run(self):
        cnt=0
        for i in range(self.until):
            time.sleep(0.01)
            # with self.num.get_lock():
            self.num.value+=1
            # print(self.num.value)
            # print('run ', self.id, self.num.value)

            # print('test')

if __name__ == '__main__':


    start_time = time.time()
    num_single = mp.Value('i', 0)
    w =Worker(0, num_single, 500)
    w.start()
    w.join()
    elapsed_time_single = time.time()-start_time

    data = []
    # data.share_memory()

    print(num_single.value)
    print(elapsed_time_single)

    start_time = time.time()
    num = mp.Value('i', 0)
    workers = [Worker(i, num, 00) for i in range(10)]
    [w.start() for w in workers]
    [w.join() for w in workers]
    elapsed_time = time.time() - start_time



    print(num.value)
    print(elapsed_time)





    # lock = Lock()
    # workers = []
    # num = mp.Value('i', 0)
    # for w in range(10):
    #     p = mp.Process(target=adder, args=(w,num, lock))
    #     p.start()
    #     workers.append(p)
    #
    # for w in workers:
    #     w.join()
    #
    # print(num.value)




    # print(mp.cpu_count())
    # q = Queue()
    # data=[5,32,4,54,1,23,6,88,64]
    # creatorP = mp.Process(target=creator, args=(data,q))
    # consumerP1 = mp.Process(target=consumer, args=(q,))
    # consumerP2 = mp.Process(target=consumer, args=(q,))
    # creatorP.start()
    # consumerP1.start()
    # consumerP2.start()
    #
    # q.close()
    # q.join_thread()
    #
    #
    # creatorP.join()
    # consumerP1.join()
    # consumerP2.join()

    # for w in range(6):
    #     p = mp.Process(target=work, args=(w, 5000, lcok))
    #     p.start()
    #     workers.append(p)
    #
    # for w in workers:
    #     w.join()
