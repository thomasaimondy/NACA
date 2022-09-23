#!/usr/bin/python
# encoding=utf-8
import datetime
import os, sys
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default=None)
args = parser.parse_args()

def execCmd(cmd):
    try:
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('%s\t 运行失败' % (cmd))

def adds(cmds, subcmds):
    for i in range(len(subcmds)):
        cmds.append(subcmds[i])


if __name__ == '__main__':
    # 是否需要并行运行
    if_parallel = True
    
    cmds=[]
    # 需要执行的命令列表; HWDB 多层实验
    # --epochs 200 --batch-size 50 --test-batch-size 50 --lr 0.0001 --topology CONV_64_5_1_2_CONV_128_5_1_2_FC_1000_FC_10 --spike-window 20 --randKill 1 --codename test --train-mode BRP --gpu 1
    model_list = [1] # 0,1,2,3,4,5
    gpus = [0]
    for ii in range(len(model_list)):
        if args.type is None:
            print('No parameters.')
            sys.exit()
        elif args.type == 'Parameter_test':
            adds(cmds, ['nohup python -u main.py --epochs 100 --batch-size 16 --test-batch-size 16 --lr 0.0001 --topology CONV_64_5_1_2_CONV_128_5_1_2_FC_1000_FC_10 --randKill 0.1 --codename test --train-mode BRP --gpu 0 --dataset tidigits --brpscale 5e-3 > log/log_{}_{} 2>&1 &'.format(args.type, gpus[ii], model_list[ii], model_list[ii], args.type)])
            break

        elif args.type == 'eprop_snn_tidigits':
            adds(cmds, ['nohup python -u main.py --epochs 100 --batch-size 16 --test-batch-size 16 --lr 0.0001 --topology CONV_64_5_1_2_CONV_128_5_1_2_RFC_1000 --spike_window 20 --randKill 0.1 --codename {} --train-mode DFA --gpu {} --seed {} --dataset tidigits > log/log_{}_{} 2>&1 &'.format(args.type, gpus[ii], model_list[ii], model_list[ii], args.type)])
        elif args.type == 'eprop_snn_mnist':
            adds(cmds, ['nohup python -u main.py --epochs 100 --batch-size 32 --test-batch-size 32 --lr 0.0001 --topology CONV_64_5_1_2_CONV_128_5_1_2_RFC_1000 --spike_window 20 --randKill 1 --codename {} --train-mode DFA --gpu {} --seed {} --dataset MNIST > log/log_{}_{} 2>&1 &'.format(args.type, gpus[ii], model_list[ii], model_list[ii], args.type)])
    
    
    for i in range(len(cmds)):
        print('[ ]'+cmds[i])
    input_ = input('一共有'+str(len(cmds))+'个进程，要继续吗？（y/n）:')
    if input_ == 'y':
        pass
    else:
        sys.exit()

    
    if if_parallel:
	    # 并行
        threads = []
        for cmd in cmds:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)
	
	    # 等待线程运行完毕
        for th in threads:
            th.join()
    else:
	    # 串行
        for cmd in cmds:
            try:
                print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
                os.system(cmd)
                print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
            except:
                print('%s\t 运行失败' % (cmd))
