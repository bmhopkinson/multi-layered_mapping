import multiprocessing as mp
import math

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_concurrent(context, func=[], data_in=[], args=[], n_workers=2):
    if context.manager is None:
        context.manager = mp.Manager()
    results = context.manager.dict()  # this will hold info about which faces are visible in which frames:  key 'faceid_frame_id', value: dist from center of projected face to camera projection center,
    # this structure is simpler to deal with in multiprocesing context and will be postproccesed later
    jobs = []
    for chunk in chunks(data_in, math.ceil(len(data_in) / n_workers)):
        j = mp.Process(target=func,
                       args=(chunk, results, args))
        j.start()
        jobs.append(j)

    for j in jobs:
        j.join()

    return results.copy()  # convert to normal dictionary

def run_singlethreaded(context, func=[], data_in=[], args=[], n_workers=1):
    ''' run a function single threaded, has same interface as run_concurrent for easy swapping'''
    results = {}
    results = func(data_in, results, args)
    return results