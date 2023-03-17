#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import json
import numpy as np

def main():
    results_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'results')
    dn_list = [dn for dn in os.listdir(results_dir) if osp.isdir(osp.join(results_dir, dn)) and osp.exists(osp.join(results_dir, dn, 'eval_results.json'))]
    results = [[] for _ in range(500)]
    for dn in dn_list:
        with open(osp.join(results_dir, dn, 'eval_results.json')) as f:
            result = json.load(f)
        if 'task' not in dn:
            task_id = 0
        else:
            tindex = dn.find('task')
            task_id = int(dn[tindex+4:dn.find('_', tindex)])
        results[task_id].append(result)

    for i, result in enumerate(results):
        if len(result) == 0:
            continue
        print('task %d' % i)
        for key in result[0].keys():
            print('%s: %.4f' % (key, np.mean([r[key] for r in result if not isinstance(r[key], str)])))

if __name__ == '__main__':
    main()

