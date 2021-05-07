# include: utf-8
import json
import multiprocessing
import os
import pandas as pd


class MultiProcess:
    """
        mp = MultiProcess()
        mp.set_queue_pool()
        func = partial(func, **kwargs)
        mp.set_job(func, all_code=pro.data_select())
        mp.start()
        mp.gather_queue_pool(save_path='result.json')

    """

    def __init__(self, num_workers=None):
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = max(int((multiprocessing.cpu_count() * 0.7)), 1)
        self.jobs = None
        self.kwargs = None
        self.parameters = dict()
        self.functions = None
        self.tasks = None
        self.queue = None
        self.queue_pool = []

    def set_queue(self, queue=None):
        if queue is None:
            self.queue = multiprocessing.Queue()
        else:
            self.queue = queue

    def get_queue(self):
        if self.queue is None:
            self.queue = multiprocessing.Queue()
        return self.queue

    def set_queue_pool(self):
        self.get_queue_pool()

    def get_queue_pool(self):
        for i in range(self.num_workers - len(self.queue_pool)):
            self.queue_pool.append(multiprocessing.Queue())

    def set_job(self, jobs, **kwargs):
        """

        :param jobs: which function need to parallel execute. e.g partial function.
        :param kwargs: which parameters need to splitting.
        :return:
        """
        self.jobs = jobs
        self.kwargs = dict(**kwargs)

    def start(self):
        for key, values in self.kwargs.items():
            if isinstance(values, pd.DataFrame):
                task_per = [[] for _ in range(self.num_workers)]
                temp = values.iterrows()
            elif isinstance(values, list):
                task_per = [[] for _ in range(self.num_workers)]
                temp = enumerate(values)
            else:
                task_per = [None for _ in range(self.num_workers)]
                temp = enumerate([values] * self.num_workers)
            for i, value in temp:
                i = i % self.num_workers
                if task_per[i] is None:
                    task_per[i] = value
                else:
                    task_per[i].append(value)
            self.parameters.setdefault(key, task_per)
        self.__run__()

    def extract_parameter(self, i):
        result = dict()
        for key in list(self.parameters.keys()):
            result.setdefault(key, self.parameters.get(key, [])[i])
        queue = None
        if self.queue is not None:
            queue = self.queue
        if len(self.queue_pool) != 0:
            queue = self.queue_pool[i]
        if queue is not None:
            result.setdefault('queue', queue)
        return result

    def __run__(self):
        jobs = []
        values_temp = self.parameters.get(list(self.parameters.keys())[0], [])
        for i in range(self.num_workers):
            if not len(values_temp):
                continue
            p = multiprocessing.Process(target=self.jobs,
                                        kwargs=self.extract_parameter(i))
            jobs.append(p)
        for p in jobs:
            p.start()
        for p in jobs:
            p.join()
        print("{} sub_multi_processing is over.".format(__name__))

    def gather_queue(self, *args, **kwargs):
        result = []
        while not self.queue.empty():
            result.append(self.queue.get(block=False))
        return result

    def gather_queue_pool(self, save_path=None):
        result = {}
        count = 0
        for queue in self.queue_pool:
            while not queue.empty():
                result.setdefault(count, queue.get(block=False))
                count += 1
        # save
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(result, f)
        return result
