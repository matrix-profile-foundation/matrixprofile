import time

import numpy as np
from matrixprofile.stomp_parallel import stomp_parallel
from matrixprofile import stomp
import ray

# ts = np.loadtxt("/home/tyler/src/matrixprofile-ts-redesign/docs/examples/rawdata.csv", skiprows=1)
# w = 32

ts = np.random.uniform(size=2**12)
w = 2**8

start = time.time()
stomp_parallel(ts, w, n_jobs=4)

loc_result = "stomp parallel local: {}".format(time.time() - start)

ray.init(num_cpus=4, ignore_reinit_error=True, logging_level=40,)
start = time.time()
stomp_parallel(ts, w)
ray_result = "ray parallel local: {}".format((time.time() - start))
ray.shutdown()

start = time.time()
stomp(ts, w)
ser_result = "stomp: {}".format(time.time() - start)

print(loc_result)
print(ray_result)
print(ser_result)