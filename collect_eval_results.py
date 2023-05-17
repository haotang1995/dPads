#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import numpy as np

def main(fast_flag=False,):
    results_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'results')
    basename = 'eval_results.json' if not fast_flag else 'fast_eval_results.json'
    dn_list = [dn for dn in os.listdir(results_dir) if osp.isdir(osp.join(results_dir, dn)) and osp.exists(osp.join(results_dir, dn, basename))]
    results = [[] for _ in range(500)]
    for dn in dn_list:
        with open(osp.join(results_dir, dn, basename)) as f:
            try:
                result = json.load(f)
            except Exception as err:
                print(dn, basename, err)
                raise err
        if 'task' not in dn:
            task_id = 0
        else:
            tindex = dn.find('task')
            task_id = int(dn[tindex+4:dn.find('_', tindex)])
        results[task_id].append(result)

    for i, result in enumerate(results):
        if len(result) == 0:
            continue
        print('Task {}: {}'.format(i, ', '.join(['%s: %.4f' % (key, np.max([r[key] for r in result])) for key in result[0].keys() if not isinstance(result[0][key], str)])))
    print()

    print('Evaluated on %d tasks' % len([result for result in results if len(result) > 0]))
    first_result = [result for result in results if len(result) > 0][0]
    for key in first_result[0].keys():
        if isinstance(first_result[0][key], str):
            continue
        print('%s: %.4f' % (key, np.mean([np.max([r[key] for r in result]) for result in results if len(result) > 0])))
    print('Missing results: ', [ri for ri, result in enumerate(results) if len(result) == 0])

if __name__ == '__main__':
    main(fast_flag=False,)
    # main(fast_flag=True,)

