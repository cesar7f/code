from multiprocessing import Process
import os

def worker():
    print("Worker process {}".format(os.getpid()))

if __name__ == "__main__":
    proc1 = Process(target=worker)
    proc1.start()
    proc2 = Process(target=worker)
    proc2.start()