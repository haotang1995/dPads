#!/usr/bin/env python
# coding=utf-8

import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle as pkl
from tqdm import tqdm
import json, random
import os, os.path as osp
from program_graph import Edge
from utils.logging import print_program
from nsp.tasks.arith.base import ArithTask

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def get_symbol_grounder(graph):
    root_node = graph.root_node
    queue = [root_node]
    cur_depth = -1
    cur_node_id = 0
    # BFS
    while len(queue) != 0:
        cur_node = queue.pop(0)
        # new depth
        if cur_node.depth > cur_depth:
            cur_depth += 1
            cur_node_id = 0
        # type sign
        for type_sign in cur_node.prog_dict:
            # program name
            for prog_id, prog in enumerate(cur_node.prog_dict[type_sign]):
                # weight and connected program
                for subm, edge in prog.get_submodules().items():
                    if isinstance(edge, Edge):
                        next_node = edge.to_node
                        next_progs = next_node.prog_dict[edge.type_sign]
                        for next_prog in next_progs:
                            if next_prog.name in 'XYZ':
                                return next_prog.cnn_layer
                        if next_node not in queue:
                            queue.append(next_node)
        # next node
        cur_node_id += 1

def run_program(program, s):
    program = program.strip()
    op = program[:program.find('(')]
    sub_prog = program[program.find('(')+1:program.rfind(')')]
    sub_arg_list = []

    cur_index, left_num = 0, 0
    while cur_index < len(sub_prog):
        if sub_prog[cur_index] == '(':
            left_num += 1
        elif sub_prog[cur_index] == ')':
            left_num -= 1
        elif sub_prog[cur_index] == ',' and left_num == 0:
            sub_arg_list.append(sub_prog[:cur_index])
            sub_prog = sub_prog[cur_index+1:]
            cur_index = 0
        cur_index += 1
    if sub_prog != '':
        sub_arg_list.append(sub_prog)

    if op == 'Add':
        assert(len(sub_arg_list) == 2)
        return run_program(sub_arg_list[0], s) + run_program(sub_arg_list[1], s)
    elif op == 'Sub':
        assert(len(sub_arg_list) == 2)
        return run_program(sub_arg_list[0], s) - run_program(sub_arg_list[1], s)
    elif op == 'Multiply':
        assert(len(sub_arg_list) == 2)
        return run_program(sub_arg_list[0], s) * run_program(sub_arg_list[1], s)
    elif op == 'Start':
        assert(len(sub_arg_list) == 1)
        return run_program(sub_arg_list[0], s)
    elif op == 'Map':
        assert(len(sub_arg_list) == 1)
        return run_program(sub_arg_list[0], s)
    elif op == 'X':
        assert(len(sub_arg_list) == 0)
        return s[..., 0:1]
    elif op == 'Y':
        assert(len(sub_arg_list) == 0)
        return s[..., 1:2]
    elif op == 'Z':
        assert(len(sub_arg_list) == 0)
        return s[..., 2:3]
    else:
        raise ValueError('Unknown op: {}'.format(op))

def eval_graph(path, train_data, test_data,):
    dn = osp.basename(path)
    if 'graph.p' not in os.listdir(path):
        return
    output_fn = osp.join(path, 'fast_eval_results.json')
    if osp.exists(output_fn):
        return

    if 'task' not in dn:
        task_id = 0
    else:
        tindex = dn.find('task')
        task_id = int(dn[tindex+4:dn.find('_', tindex)])

    with open(osp.join(path, 'graph.p'), 'rb') as f:
        graph = pkl.load(f)
    grounder = get_symbol_grounder(graph).to(device)
    program = print_program(graph.extract_program())
    print(dn, program, task_id)

    x, s, y = train_data
    x, s, y = x.to(device), s.to(device), y.to(device)
    y = y[..., task_id:task_id+1]

    pred_s = grounder(x)
    assert(pred_s.shape == s.shape)
    sym_acc = torch.mean(((pred_s*10).long() == (s*10).long()).float()).item()

    pred_y = run_program(program, s)
    assert(pred_y.squeeze().shape == y.squeeze().shape)
    prog_acc = pred_y.squeeze().allclose(y.squeeze(), atol=1e-5)

    pred_y = run_program(program, pred_s)
    assert(pred_y.squeeze().shape == y.squeeze().shape)
    loss = F.mse_loss(pred_y, y).item()

    test_x, test_s, test_y = test_data
    test_x, test_s, test_y = test_x.to(device), test_s.to(device), test_y.to(device)
    test_y = test_y[..., task_id:task_id+1]

    test_pred_s = grounder(test_x)
    assert(test_pred_s.shape == test_s.shape)
    test_sym_acc = torch.mean(((test_pred_s*10).long() == (test_s*10).long()).float()).item()

    test_pred_y = run_program(program, test_s)
    assert(test_pred_y.squeeze().shape == test_y.squeeze().shape)
    test_prog_acc = test_pred_y.squeeze().allclose(test_y.squeeze(), atol=1e-5)

    test_pred_y = run_program(program, test_pred_s)
    assert(test_pred_y.squeeze().shape == test_y.squeeze().shape)
    test_loss = F.mse_loss(test_pred_y, test_y).item()

    print('Test Symbol Acc: {}, Test Program Acc: {}, Test Loss: {}'.format(test_sym_acc, test_prog_acc, test_loss))

    with open(output_fn, 'w') as f:
        json.dump({
            'task_id': task_id, 'program': program, 'symbol_acc': sym_acc, 'program_acc': prog_acc, 'loss': loss,
            'test_symbol_acc': test_sym_acc, 'test_program_acc': test_prog_acc, 'test_loss': test_loss,
        }, f)

def get_data():
    task_config = {'task_num': 500, 'random_task_flag': False, 'dataset_size_str': 'normal', 'dataset_size': 1000, 'test_dataset_size': 1000, 'symbol_num': 3, 'image_dataset_name': 'cifar10', 'split_image_flag': True, 'norm10_flag': True}
    # task_config = {'task_num': 500, 'random_task_flag': False, 'dataset_size_str': 'small', 'test_dataset_size': 1000, 'symbol_num': 3, 'image_dataset_name': 'cifar10', 'split_image_flag': True, 'norm10_flag': True}
    task = ArithTask(**task_config)
    target_formula_list = task.get_target_formula_list()
    dataset = task.build_dataset(program_list=target_formula_list, train_flag=True, dry_run=False,)
                                 # io_pairs_list=torch.tensor(list(itertools.product(list(range(10)), repeat=3)), dtype=torch.long,),)
    test_dataset = task.build_dataset(program_list=target_formula_list, train_flag=False, dry_run=False,)
    return dataset[:], test_dataset[:]

def main():
    data_loader, test_data = get_data()
    results_dir = osp.abspath(osp.join(osp.dirname(__file__), 'results'))
    dn_list = os.listdir(results_dir)
    random.shuffle(dn_list)
    for dn in dn_list:
        if 'imgmath' not in dn:
            continue
        with torch.no_grad():
            eval_graph(osp.join(results_dir, dn), data_loader, test_data)

if __name__ == '__main__':
    main()
