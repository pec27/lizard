"""
Parallel utilities for Lizard, in particular using pythons multiprocessing
module to allow tasks to be shared out over a pool of processes.

Also includes a combination-logger so that all jobs can write to the same log 
(can obviously still cause problems if they write a lot).

Peter Creasey - Feb 2016
"""
from __future__ import print_function, division, unicode_literals, absolute_import
import sys
from .log import VerboseTimingLog, PrefixedIO, LockedLineIO, MarkUp
from . import domain
from . import p3m
import pickle
from time import time
import signal
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
import numpy as np


_myproc = None # To hold process data, initialised in init_worker()
_free_connections = None

class MyProc:
    def __init__(self, log, lock, root_time, connections):
        self.log = log # the log
        self.lock = lock
        self.root_time = root_time
        self.conn = connections # end of pipe - TODO have more than one so that processes can talk individually


def get_free_connection():
    """ 
    Get the index of a connection in order to talk to workers - i.e. pass in
    the returned value to the worker so you can talk on the same pipe.
    """

    if len(_free_connections)==0:
        raise Exception('No free connections. Perhaps you forgot to release one?')
    print('Current free connections', _free_connections, file=_myproc.log)

    idx = _free_connections.pop()
    return idx

def release_connection(conn):
    idx = _myproc.conn.index(conn)
    _free_connections.append(idx)


def get_log():
    return _myproc.log

def init_worker(lock, proc_counter, root_time, log_file_name, connections):
    """
    Initialise this process as a slave, using arguments passed from the master,
    mostly just to allow writing to the same log. To fix some of the broken
    exception handling in multiprocessing we make the workers ignore keyboard
    interrupt exceptions.

    lock          - a Lock object (e.g. so we can write to the log)
    proc_counter  - a Value object that we find our ID from and update
    root_time     - time.time() on the master, in case we need to write the
                    relative time
    log_file_name - name/None (i.e. sys.stdout)
    connections   - 1 pipe per proc so we can send intermediate results back to
                    the master
    """
    # Make the workers (sub-processes) ignore KeyboardException, the main process 
    # will handle them (killing off the workers)
    signal.signal(signal.SIGINT, signal.SIG_IGN) 
    t0 = time()
    with lock:
        my_proc = proc_counter.value + 1
        proc_counter.value = my_proc

    if log_file_name is None:
        buf = sys.stdout
    else:
        buf = open(log_file_name, 'a')

    # Acquire a lock before writing a line, indent to the correct amount
    buf = PrefixedIO(LockedLineIO([buf],lock), str(' '*(8*my_proc)))

    conn = connections[0]

    with lock:
        conn.send(MarkUp.HEADER+'Proc %d testing communication'%my_proc+MarkUp.ENDC)

    log = VerboseTimingLog(bufs=[buf], also_stdout=False, insert_timings=True)
    global _myproc
    _myproc = MyProc(log, lock, root_time, connections)

    print(MarkUp.OKBLUE+'Starting proc', my_proc, 'at time %.3f'%(t0-root_time)+MarkUp.ENDC, file=log)


def build_pool(nproc, log_file_name=None):
    """
    Make a multiprocessing pool object and a shared log (either a file or 
    stdout) object. 

    nproc - number of processes
    log_file_name - name/None

    returns pool, log
    """
    print('Building pool with', nproc, 'processors')
    lock = mp.Lock() # a lock we can prevent collisions with
    proc_counter = mp.Value('i', 0)

    if log_file_name is None:
        buf = sys.stdout
    else:
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        buf = open(log_file_name, 'a')

    # Acquire a lock before writing a line
    buf = LockedLineIO([buf], lock)
    log = PrefixedIO(buf, str(' '*1))
    log = VerboseTimingLog(bufs = [log], also_stdout=False, insert_timings=True)

    # make as many pipes as processors (pass all-to-all, will dynamically use)
    connections = [mp.Pipe() for i in range(nproc)]
    slave_connections = [c[1] for c in connections]
    my_connections = [c[0] for c in connections]

    t = time()

    pool = mp.Pool(nproc, initializer=init_worker, initargs=(lock, proc_counter, t, log_file_name, slave_connections))
    for i in range(nproc):
        obj = my_connections[0].recv()
        print(obj, file=log)
    print('Built worker pool with {:,} processors'.format(proc_counter.value), file=log)

    global _myproc
    _myproc = MyProc(log, lock, t, my_connections)
    global _free_connections
    _free_connections = list(range(nproc))

    # prevent exceptions on teardown by cleaning up logs
    def cleanup_log():
        """ Cleanup the logs """
        get_log().close()

    Finalize(None, cleanup_log, exitpriority=16)
    return pool, log


class PooledShort:
    """ 
    The short-force calculated over a pool of processes 
    
    TODO at some point would prefer to pass in **kwargs for get_force_split
    so we dont need to know about r_soft, deconvolve, kernel_pts etc.
    """
    def __init__(self, nproc, wts, pos, r_soft, r_split, pool, log):

        self._log = log
        print('Creating domain indices', file=log)
        domains = list(domain.sq_wtd_voxels(nproc, pos, r_split, log=log))
        self._domains = domains
        self._accel = np.empty_like(pos)
        
        self._results = pp_distrib(pool, domains, wts, pos, r_soft, r_split, log)

    def get(self):
        """ Blocking call to get all the particle-particle accelerations """
        accel = self._accel
        for r in self._results:
            job, acc, t_sent = pickle.loads(r)
            print('Received {:,} accelerations'.format(len(acc)), 
                  '(job %d) in %.3fs'%(job, time()-t_sent), file=self._log)
            idx_domain, idx_non_ghosts = self._domains[job]
            accel[idx_domain[idx_non_ghosts]] = acc

        return accel

def pp_distrib(pool, domains, wts, pos, r_soft, r_split, log):
    """
    Distribute a particle-particle calculation over the pool of processors,
    returns a pool.imap_unordered object that gives a sequence of results 
    for each domain.
    """
    # Pickle is very space inefficient with Numpy arrays, need to specify
    # HIGHEST_PROTOCOL to actually pass doubles in a normal way (i.e. 8 bytes per value)
    protocol = pickle.HIGHEST_PROTOCOL
    # if wts is an array then we index for the given domain, otherwise same 
    # scalar for all
    if np.isscalar(wts):
        args = iter(pickle.dumps((job, idx_non_ghosts, wts, pos[idx], r_soft, r_split, time()),
                                  protocol=protocol)
                    for job, (idx, idx_non_ghosts) in enumerate(domains))
    else:
        args = iter(pickle.dumps((job, idx_non_ghosts, wts[idx], pos[idx], r_soft, r_split, time()),
                                  protocol=protocol)
                    for job, (idx, idx_non_ghosts) in enumerate(domains))
        

    print('Serialising and launching jobs', file=log)
    try:
        results = pool.imap_unordered(pp_job, args)
        
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers", file=log)
        pool.terminate()
        pool.join()
        raise
    return results
        
def pp_job(args):
    """ An individual particle-particle force job """
    t_started = time()
    root_time = _myproc.root_time
    t0 = time() - root_time
    log = get_log()
    try:
        print(MarkUp.HEADER+'<PP job started at time %.3fs>'%t0+MarkUp.ENDC, file=log)
        job, non_ghosts, wts, pos, r_soft, r_split, t_sent = pickle.loads(args)
        print('Time taken to communicate %.3fs'%(t_started-t_sent), file=log)
        fs = p3m.get_force_split(r_split, mode='erf')
        pairs, accel = p3m.pp_accel(fs, wts, pos, r_soft=r_soft, log=log)
        t_sent = time()
        print('Serialising {:,} accelerations'.format(len(non_ghosts)),file=log)
        res = pickle.dumps((job, accel[non_ghosts], t_sent), pickle.HIGHEST_PROTOCOL)
        t1 = time() - root_time
        print('<PP job stopped at time %.3f>, sleeping'%t1,file=log)
    except Exception as e:
        print(e, file=log)
        raise
    return res
