from multiprocessing import Lock, Pool

def init_pool_processes(the_lock):
    '''Initialize each process with a global variable lock.
    '''
    global lock
    lock = the_lock

class Test:
    def function(self, i):
        lock.acquire()
        with open('test.txt', 'a') as f:
            print(i, file=f)
        lock.release()
    def anotherfunction(self):
        lock = Lock()
        pool = Pool(initializer=init_pool_processes, initargs=(lock,))
        pool.map(self.function, range(10))
        pool.close()
        pool.join()

if __name__ == '__main__':
    t = Test()
    t.anotherfunction()
