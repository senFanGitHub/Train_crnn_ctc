import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import mxnet as mx
from common import crnn_trainer,crnn_infer
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='1'

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-train_root", "--train_root", action="store", dest="train_root", default='', 
    help="训练集图像文件夹", required=True)
parser.add_argument("-max_len", "--max_len", action="store", dest="max_len", default=4, 
    help="训练集中单张图片字母的最大长度", required=True)
parser.add_argument("-test_root", "--test_root", action="store", dest="test_root", default='', 
    help="测试集图像文件夹", required=True)
parser.add_argument("-vocab", "--vocab_list", action="store", dest="vocab_list",default='0123456789qazwsxedcrfvtgbyhnujmikolp+-*/QAZWSXEDCRFVTGBYHNUJMIKOLP', help="字符分类类别列表", required=False)
 
parser.add_argument("-max_update", "--max_update", action="store", dest="max_update", default=8000, 
    help="最大学习步数", required=False)

parser.add_argument("-lstm_hid", "--lstm_hid", action="store", dest="lstm_hid", default=64, 
    help="crnn里bi-lstm中间向量维度", required=False)
parser.add_argument("-backbone", "--backbone", action="store", dest="backbone", default="ResNetV2_large", 
    help="crnn里cnn的backbone类型", required=False)
parser.add_argument("-model_dir", "--model_dir", action="store", dest="model_dir", default="model_crnn", 
    help="存储模型文件夹", required=False)
parser.add_argument("-batch_size", "--batch_size", action="store", dest="batch_size", default=32, 
    help="batch size", required=False)
parser.add_argument("-concat_num", "--concat_num", action="store", dest="concat_num", default=1, 
    help="慎选，concat_num=1：正常，concat_num=n：随机拼接n张当做一张长的。拼接方式仅在部分字母分布均匀的验证码场景适用", required=False)
parser.add_argument("-s_lr", "--s_lr", action="store", dest="s_lr", default=0.01,
    help="初始学习率", required=False)
parser.add_argument("-e_lr", "--e_lr", action="store", dest="e_lr", default=0.0001,
    help="结束时的学习率", required=False)# 采用的cosine学习率调整策略
parser.add_argument("-c_e_r", "--char_embed_ratio", action="store", dest="char_embed_ratio", default=2,
    help="平均每个字符所占特征序列数", required=False)# 采用的cosine学习率调整策略



parser.print_help()  
args = parser.parse_args()
'../金融许可train'
def Do_train():
    Train_tool = crnn_trainer(train_root=args.train_root,test_root=args.test_root,
                              lstm_hid=args.lstm_hid,backbone=args.backbone,
                              model_dir=args.model_dir,  concat_num=args.concat_num,
                              batch_size = args.batch_size,s_lr=args.s_lr,e_lr=args.e_lr,
                              max_len=args.max_len,max_update=args.max_update,vocab_list=args.vocab_list)
    Train_tool.do_training()
    
def Do_speed_eval():
    import time
    cpu_infer =crnn_infer(args.model_dir,ctx=mx.cpu())
    npimg=np.ones((32,128,3))# any img shape,will be resized
    for i in range(50):#warm up
        cpu_infer.predict(npimg)
    A=time.time()
    print("wait about 1 min")
    for i in range(1000):

        cpu_infer.predict(npimg)
    B=time.time()
    print(cpu_infer.predict(npimg))
    backbone = cpu_infer.info['backbone']
    print(f'{backbone} cpu mean cost per img:{(B-A)/1000}')
    
if __name__ == "__main__":
    
    Do_train()
    Do_speed_eval()