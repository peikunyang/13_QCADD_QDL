import sys
import torch
import math
import numpy as np
import pennylane as qml
import datetime
from torch import optim
from Global import *
from Read_Data.Read_Data import *
from Read_Data.Layer import *

def Train_Pyt(bfe,img):
  sam=bfe.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  random.shuffle(num)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    bfe2=bfe[num[beg_frm:end_frm]]
    img1=img[num[beg_frm:end_frm]]
    img2=100*Circ_Tor_Ful9(img1.cfloat(),Par,batch).reshape(batch,2)
    pred1=img2[:,0]
    pred2=img2[:,1]
    pred=pred1-pred2
    err=pred-bfe2
    loss=torch.mul(err,torch.conj(err)).sum(0)
    Opt.zero_grad()
    loss.backward()
    Opt.step()

def Train(n_batch):
  num=np.arange(len(Pdb))
  random.shuffle(num)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*batch
    end_frm=(i+1)*batch
    if end_frm>len(Pdb):
      end_frm=len(Pdb)
      batch=end_frm-beg_frm
    bfe=Bfe[num[beg_frm:end_frm]]
    img=Img[num[beg_frm:end_frm]]
    Train_Pyt(img,bfe,batch)

def Pred_Pyt(img,par):
  E=[]
  sam=img.shape[0]
  n_batch=int(np.ceil(sam/Batch))
  num=np.arange(sam)
  batch=Batch
  for i in range (n_batch):
    beg_frm=i*Batch
    end_frm=(i+1)*Batch
    if end_frm>sam:
      end_frm=sam
      batch=end_frm-beg_frm
    img1=img[num[beg_frm:end_frm]]
    img2=100*Circ_Tor_Ful9(img1.cfloat(),Par,batch).reshape(batch,2)
    pred1=img2[:,0]
    pred2=img2[:,1]
    pred=(pred1-pred2).tolist()
    E=E+pred
  return E

def Pred_Pen(img,par):
  E=[]
  for i in range(img.shape[0]):
    pro=100*Circ_Pen_Ful9(img[i],par).reshape(-1,2)
    pred1=pro[:,0]
    pred2=pro[:,1]
    pred=(pred1-pred2).item()
    E.append(pred)
  return E

def Main():
  global Par,Pdb,Bfe,Img,Opt
  Par=Para()
  Pdb_gen,Pdb_ref,Bfe_gen,Bfe_ref=Read_PDB()
  Img_gen=Read_Img(Pdb_gen)
  Img_ref=Read_Img(Pdb_ref)
  fw=open("Result/rmsd","w", buffering=1)
  Opt=optim.SGD([Par],lr=learning_rate)
  for i in range (N_Ite):
    Train_Pyt(Bfe_gen,Img_gen)
    if i%10==9:
      E_gen_pyt=Pred_Pyt(Img_gen,Par.detach())
      Out_E_diff(fw,i,Bfe_gen,E_gen_pyt)
  E_gen_pyt=Pred_Pyt(Img_gen,Par.detach())
  E_gen_pen=Pred_Pen(Img_gen,Par.detach().to('cpu'))
  E_ref_pyt=Pred_Pyt(Img_ref,Par.detach())
  E_ref_pen=Pred_Pen(Img_ref,Par.detach().to('cpu'))
  Out_Energy('E_train',Bfe_gen,E_gen_pyt,E_gen_pen)
  Out_Energy('E_test',Bfe_ref,E_ref_pyt,E_ref_pen)
  Out_E_diff(fw,N_Ite,Bfe_gen.to('cpu'),E_gen_pen)
  Out_E_diff(fw,N_Ite,Bfe_ref.to('cpu'),E_ref_pen)
  OutPara(Par.detach())
  fw.close()

start=datetime.datetime.now()
Main()
end=datetime.datetime.now()
print("執行時間：",end-start)

