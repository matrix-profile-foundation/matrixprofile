from datetime import datetime

import numpy as np
from matrixprofile.stomp_parallel import stomp_parallel
from matrixprofile import stomp

#ts = np.loadtxt("/home/tmarrs/src/matrixprofile-ts-redesign/docs/examples/rawdata.csv", skiprows=1)
#w = 32

ts = np.random.uniform(size=2**12)
w = 2**5

start = datetime.now()
stomp_parallel(ts, w, use_ray=False)
print((datetime.now() - start), 'stomp parallel local')

import ray

ray.init(num_cpus=4, ignore_reinit_error=True, logging_level=40,)
start = datetime.now()
stomp_parallel(ts, w, use_ray=True)
print((datetime.now() - start), 'stomp parallel ray')
ray.shutdown()

start = datetime.now()
stomp(ts, w)
print((datetime.now() - start), 'stomp')
