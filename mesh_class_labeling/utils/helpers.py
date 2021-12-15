import multiprocessing as mp
import math
import numpy as np


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_concurrent(context, func, data_in, args=None, n_workers=2):
    """concurrently run a function (func) on data (data_in) with arguments (args) in a context (e.g. object that owns
        the function). n_workers specifies number of multiprocessing jobs to start

        func must have the following interface:
        func(chunk_of_data_in, results_dict, optional_args)
        """

    if context is None:
        manager = mp.Manager()
    elif context.manager is None:
        context.manager = mp.Manager()
        manager = context.manager
    else:
        manager = context.manager

    results = manager.dict()    # results dictionary - note b/c of python multiprocessing constraints only option is
                                        # to dump data into this dictionary, different processes aren't notified when it is modified
    jobs = []
    for chunk in chunks(data_in, math.ceil(len(data_in) / n_workers)):
        j = mp.Process(target=func,
                       args=(chunk, results, args))
        j.start()
        jobs.append(j)

    for j in jobs:
        j.join()

    return results.copy()  # convert to normal dictionary


def run_singlethreaded(context, func, data_in, args=None, n_workers=1):
    """ run a function single threaded, has same interface as run_concurrent for easy swapping"""
    results = {}
    results = func(data_in, results, args)
    return results


def collate_results(raw_ds, key_parser):
    """takes raw dictionaries with multiple observations per face (different frames) and collates them by face,
    uses key_parser to split keys of raw dictionary into new desired higher level grouping. for example raw dict
    may have keys as 'frameid_faceid' and key_parser splits off frame_id as the new grouping """
    collated_ds = {}
    for instance in raw_ds:
        m = key_parser.search(instance)
        new_key = int(m.group(1))
        if new_key in collated_ds:
            collated_ds[new_key] = np.append(collated_ds[new_key], np.array(raw_ds[instance], ndmin=2), axis=0)
        else:
            collated_ds[new_key] = np.array(raw_ds[instance], ndmin=2)

    return collated_ds

