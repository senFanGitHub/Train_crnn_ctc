import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from backbones.custome_DenseNet import DenseNet
from backbones.custome_MobileNetV2 import MobileNetV2_small,MobileNetV2_mid,MobileNetV2_large
from backbones.custome_ResNetV2 import ResNetV2_small,ResNetV2_mid,ResNetV2_large
import mxnet as mx
from mxnet import nd, autograd, gpu, cpu, gluon
from mxnet.gluon import data
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import random
from tqdm import *
import json 
import time
backbone_choices={"MobileNetV2_small":MobileNetV2_small,
                  "MobileNetV2_mid":MobileNetV2_mid,
                  "MobileNetV2_large":MobileNetV2_large,
                  "ResNetV2_small":ResNetV2_small,
                  "ResNetV2_mid":ResNetV2_mid,
                  "ResNetV2_large":ResNetV2_large,
                  "DenseNet":DenseNet}


class crnn_data(data.Dataset):
    def __init__(self,root,vocab2index,aug=False,max_len = 4,h=32,scale=16,concat_num=1,format='.jpg',char_embed_ratio=2.):
        """
        scale: 对应crnn的特征图vs原图的缩放比例
        concat_num：随机将concat_num张验证码拼接成一张作为训练对象（仅适用于字母在图上分布相对均匀的情况）
        max_len：训练集中验证码长度最多的字数
        """
        self.vocab2index=vocab2index
        self.format=format
        self.concat_num=concat_num
        self.aug=aug
        jpgs =  os.listdir(root)
        self.jpgs =[os.path.join(root,p) for p in jpgs if self.format in p ]
        self.scale = scale
        self.h=h
        self.seq_len = int(float(max_len)*char_embed_ratio)
        print(f"该数据集的图像经过cnn后会沿水平抽取出{self.seq_len}个特征向量，请确保每个字符的宽度大于图像宽度/{self.seq_len}，否则信息可能丢失，需调大char_embed_ratio")
        self.input_size=(self.seq_len*self.scale,self.h)

    def __len__(self):
        return len(self.jpgs)
    
    def get_label(self,buf):
        ret = np.ones(self.seq_len*self.concat_num)*-1 #35
        for i in range(len(buf)):
            ret[i] = int(buf[i])
        return ret
    
    def get_oneimg(self,idx):
        jpgname = self.jpgs[idx]
        img = cv2.imread(jpgname)
        try:
            img =cv2.resize(img,self.input_size)
        except:
            print(jpgname)
            print('empt file')
            os.remove(jpgname)
#         if self.aug:
#             img=numpy_img_aug.forward(img)

        string = jpgname.split(self.format)[0].split('/')[-1]
#         print(string)
        
        
        labels = [self.vocab2index[c] for c in string]

        
        img=np.multiply(img,1/255.)
        nd_img = nd.array(img.transpose(2, 0, 1))

        return nd_img ,labels
    
    def __getitem__(self,idx):
        imgs=[]
        labels=[]
        
        img1 , labels1 = self.get_oneimg(idx)
        imgs.append(img1)
        labels.extend(labels1)
        for i in range(self.concat_num-1):
            tempimg , templabels = self.get_oneimg(random.randint(0,self.__len__()-1))
            imgs.append(tempimg)
            labels.extend(templabels)
            
        
        img = nd.concat(*imgs,dim=-1)
        label = nd.array(   self.get_label(labels),dtype='int32')
        return img,label
        
        
        
        
        
        
class crnn_trainer(object):
    
    def __init__(self,train_root,test_root,ctx=mx.gpu(),pretrained='none',backbone='MobileNetV2',model_dir='model_crnn',
                 vocab_list='0123456789qazwsxedcrfvtgbyhnujmikolp+-*/QAZWSXEDCRFVTGBYHNUJMIKOLP',concat_num=1,max_len=4,batch_size=32,
                 max_update=10000,s_lr=0.005,e_lr=0.0001,char_embed_ratio=2.,lstm_hid=64):
        """
        crnn最后的类别数是字典数加1，包含“-”
        """
        #experiment parms:
        self.backbone=backbone
        self.lstm_hid = lstm_hid
        self.model_dir=model_dir
        self.concat_num=concat_num
        self.pretrained=pretrained
        self.max_update=max_update
        self.batch_size=batch_size
        self.max_len=max_len
        self.char_embed_ratio=char_embed_ratio
        self.ctx=ctx
        self.cls_num=len(vocab_list)
        self.imgs_roots=(train_root,test_root)
        self.s_lr ,self.e_lr= s_lr,e_lr
        self.scale=-1#depend on backbone,calculated during _build_model()
        
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            
        self._build_vocab(vocab_list)
        self._build_model()
        self._build_dataloader()
        self._build_opt()
    
        self.record_experiment()

    
    def record_experiment(self):
        record_type=[type(1),type(1.0),type({}),type('string'),type((1,1))]
        experiment_info={}
        for item in self.__dict__:
            A=getattr(self, item)
            if type(A) in record_type:
                experiment_info[item]=A
        with open(f'{self.model_dir}/experiment_info.json', 'w') as f:
            json.dump(experiment_info, f,ensure_ascii=False)
#         experiment_info={[backbone:self.backbone,concat_num = ,"batch_size"=32,"max_update":10000,"start_lr":0.003,"end_lr":0.0001]}
        
    
    def _build_vocab(self,vocab_list):
        print('_build_vocab')
        self.vocab2idx={}
        self.idx2vocab={}
        for i,c in enumerate(vocab_list):
            self.vocab2idx[c]=str(i)
            self.idx2vocab[i]=c
        
    def _build_model(self):
        print('_build_model')
        try:
            CRNN=backbone_choices[self.backbone]
        except:
            print('use follow:')
            print(backbone_choices)
#         self.crnn = DenseNet(classes=self.cls_num+1,num_init_features=16, growth_rate=4,lstm_hid=64)
        self.crnn = CRNN(classes=self.cls_num+1,lstm_hid=self.lstm_hid)
        
        self.crnn.collect_params().initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=self.ctx)
        if self.pretrained!='none':
            print("load from ckpt")
            self.crnn.load_parameters(self.pretrained, ctx=self.ctx,allow_missing=True,ignore_extra=True)
        
        random_input = nd.ones((1,3,32,320),ctx=self.ctx)
        out =self.crnn.features(random_input)
        scale =random_input.shape[-1]//out.shape[-1]
        self.scale=scale
        print(random_input.shape)
        print(out.shape)
        print(f'backbone feature scale:{scale}')
        
#         random_input = nd.ones((1,3,self.scale*2,320),ctx=self.ctx)
#         Ta=time.time()
#         for i in range(100):#warm up
#             out =self.crnn(random_input)
#         for i in range(200):
#             out =self.crnn(random_input)
#         Tb=time.time()
#         print(f"speed:{(Tb-Ta)/200}")
  
    def _build_dataloader(self):
        print('_build_dataloader')
        (train_root,test_root)=self.imgs_roots
        
        self.train_set =  crnn_data(train_root,self.vocab2idx,max_len=self.max_len,h=self.scale*2,concat_num=self.concat_num,scale=self.scale,char_embed_ratio=self.char_embed_ratio)
        self.TrainIter = data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, last_batch="discard", num_workers = 4)
        
        self.test_set =  crnn_data(test_root,self.vocab2idx,max_len=self.max_len,concat_num=1,h=self.scale*2,scale=self.scale,char_embed_ratio=self.char_embed_ratio)
        self.TestIter = data.DataLoader(self.test_set, batch_size=32, shuffle=True, last_batch="keep", num_workers = 2)
        self.input_size =self.test_set.input_size
        self.seq_len=self.test_set.seq_len
        
    def _build_opt(self):
        self.ctcloss = gluon.loss.CTCLoss()
        schedule = mx.lr_scheduler.CosineScheduler(base_lr=self.s_lr,final_lr=self.e_lr,max_update=self.max_update,warmup_steps=100)
        sgd_optimizer = mx.optimizer.SGD(learning_rate=self.s_lr,momentum=0.9,wd=0.0005 ,lr_scheduler=schedule)
        self.optimizer = gluon.Trainer(self.crnn.collect_params(),optimizer=sgd_optimizer)
        self.best_ac=0
        self.no_improve=0
    
    
    def check_train_iter(self):
        """
        可以通过该函数验证数据准备过程是否正确
        """
        for i, (imgs, label) in enumerate(self.TrainIter):
            img = imgs[0].asnumpy()
            img = img.transpose(1, 2, 0)    
            print(img.shape)
            img = np.multiply(img, 255.0)
            img = np.uint8(img)
#             plt.imshow(img)
#             plt.show()
            label_data =label.asnumpy()[0].tolist()
            GT = self._remove_blank(label_data)
            print(label_data)
            words = []
            for w in GT:
                words.append(self.idx2vocab[int(w)])
            string = ''.join(words)
            
            cv2.imwrite(f'{string}.jpg',img)
            break
           
    def _ctc_label(self,p):
        ret = []
        p1 = [self.cls_num] + p
        for i in range(len(p)):
            c1 = p1[i]
            c2 = p1[i + 1]
            if c2 == self.cls_num or c2 == c1:
                continue
            ret.append(c2)
        return ret

    def _remove_blank(self,l):
        ret = []
        for i in range(len(l)):
            if l[i] == -1.:
                break
            ret.append(l[i])
        return ret
    
    def Accuracy(self,label,pred):
#         SEQ_LENGTH = seq_len#35
  
        hit = 0.
        total = 0.
        rp = nd.argmax(pred, axis=2).asnumpy()
  
        for i in range(label.shape[0]):
            l = self._remove_blank(label[i].asnumpy())
            p = []
            for k in range(self.seq_len):
                  p.append(rp[i][k])
            p = self._ctc_label(p)
            if len(p) == len(l):
                match = True
                for k in range(len(p)):
                     if int(p[k]) != int(l[k]):
    #                     if (max(int(p[k]),int(l[k])) - min(int(p[k]),int(l[k])) )!=26:
                        match = False
                        break
                if match:
                     hit += 1.0
            total += 1.0

        return hit / total
    
    def evaluate_accuracy(self):

        Accuracy_sum = 0
        IT_Len = len(self.TestIter)
        for i, (imgs, label) in enumerate(self.TestIter):
            imgs = imgs.as_in_context(self.ctx)
            label = label.as_in_context(self.ctx)
            output = self.crnn(imgs)
            nd.waitall()
            batch_a =self.Accuracy(label, output)
            Accuracy_sum =Accuracy_sum+batch_a

        Ac = Accuracy_sum /IT_Len
        return Ac
    
    
    def do_training(self):
        plt_loss=[]
        plt_ac=[]
        smoothing_constant = .01
        epochs = self.max_update//len(self.TrainIter)+1
        early_stop=False
        check_every_k_epoch = max(1,200//len(self.TrainIter))
        for e in range(epochs):
        
            t1 = time.time()
            pbar = tqdm(range(len(self.TrainIter)))
            for i, (imgs_data, label_data) in zip(pbar,self.TrainIter):
                imgs_data = imgs_data.as_in_context(self.ctx)
                label_data = label_data.as_in_context(self.ctx)
                with autograd.record():
                    output = self.crnn(imgs_data)
                    loss = self.ctcloss(output, label_data)


                loss.backward()
                self.optimizer.step(imgs_data.shape[0])
                
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                               else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

                pbar.set_description(f"epo:{e}/{epochs},loss: {moving_loss}")
                
            t2 = time.time()

            if e%check_every_k_epoch==0:
                ac=self.evaluate_accuracy()
                if ac>0.1:
                    plt_ac.append(ac)
                    plt_loss.append(moving_loss)
                    nor_loss=[ls/max(plt_loss) for ls in plt_loss ]
                    if len(plt_ac)>3:
                        
                        X=list(range(len(plt_ac)))
                        plt.plot(X,nor_loss,label="loss")
                        plt.plot(X,plt_ac,label="acc")
                        plt.legend(loc = 'upper right')
                        plt.savefig(f'{self.model_dir}/train_status.jpg',dpi=100)
                        plt.clf()
                if ac>self.best_ac:
                    self.no_improve=0
                    self.best_ac=ac
                    self.crnn.save_parameters(f"{self.model_dir}/demo.params")
                    print(f'find new best ac :{self.best_ac}')
                    self.record_experiment()
                else:
                    self.no_improve+=1
                    print(f'ac:{ac}')
            if self.no_improve>5 and (self.best_ac>0.5):
                print('early stop')
                early_stop=True
                break
        if not early_stop:
            print('suggest train longer ')
            
            
            

class crnn_infer(object):
    
    def __init__(self,model_dir,ctx):
        self.ctx=ctx
        files = [os.path.join(model_dir,i) for i in os.listdir(model_dir)]
        info_json=[i for i in files if 'json' in i][0]
        self.weight = [i for i in files if 'param' in i][0]
        self._restore(info_json)

        
    def _restore(self,jsonf):
        with open(jsonf, 'r') as f:
            self.info = json.load(f)
            
        self.idx2vocab=self.info['idx2vocab']
        CRNN=backbone_choices[self.info['backbone']]
        self.input_size=tuple(self.info['input_size'])
        self.cls_num=self.info['cls_num']
        self.crnn = CRNN(classes=self.cls_num+1,lstm_hid=self.info['lstm_hid'])
        self.crnn.load_parameters(self.weight, ctx=self.ctx)
        
    def _ctc_label(self,p):
        ret = []
        p1 = [self.cls_num] + p
        for i in range(len(p)):
            c1 = p1[i]
            c2 = p1[i + 1]
            if c2 == self.cls_num or c2 == c1:
                continue
            ret.append(c2)
        return ret
    
    def preprocess(self,img):
        img=cv2.resize(img,self.input_size)
        new_img = np.multiply(img, 1 / 255.0)
        new_img = new_img.transpose(2, 0, 1)
        in_put = nd.array([new_img],ctx=self.ctx)
        return in_put

    def post_process(self,pre_distribution):
        cls = nd.argmax(pre_distribution, axis=2).asnumpy().tolist()  
        Pred=self._ctc_label(cls[0])
        words = []
        for w in Pred:
            words.append(self.idx2vocab[str(int(w))])

        return(''.join(words))
    
    def predict(self, img):
        in_put = self.preprocess(img)
        pre_distribution = self.crnn(in_put)
        result = self.post_process(pre_distribution)
        return result