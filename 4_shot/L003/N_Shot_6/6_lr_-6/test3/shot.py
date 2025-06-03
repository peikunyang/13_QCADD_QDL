import sys
import torch
import math
import numpy as np
import pennylane as qml
from Global import *
from Read_Data.Read_Data import *
from Read_Data.Layer import *

def Pred_Pen(img,par):
  E=[]
  for i in range(img.shape[0]):
    samples=Circ_Pen_Ful9(img[i],par)
    counts = torch.bincount(samples, minlength=2)
    pro = 100*counts.float() / samples.numel()
    pred1=pro[0]
    pred2=pro[1]
    pred=(pred1-pred2).item()
    E.append(pred)
  return E

def Main():
  global Par,Pdb,Bfe,Img
  Par=Read_Par()
  Pdb_gen,Pdb_ref,Bfe_gen,Bfe_ref=Read_PDB()
  Img_gen=Read_Img(Pdb_gen)
  Img_ref=Read_Img(Pdb_ref)
  fw=open("Result/rmsd","w", buffering=1)
  E_gen_pen=Pred_Pen(Img_gen,Par.detach().to('cpu'))
  E_ref_pen=Pred_Pen(Img_ref,Par.detach().to('cpu'))
  Out_Energy('E_train',Bfe_gen,E_gen_pen)
  Out_Energy('E_test',Bfe_ref,E_ref_pen)
  Out_E_diff(fw,N_Ite,Bfe_gen.to('cpu'),E_gen_pen)
  Out_E_diff(fw,N_Ite,Bfe_ref.to('cpu'),E_ref_pen)
  fw.close()

Main()

