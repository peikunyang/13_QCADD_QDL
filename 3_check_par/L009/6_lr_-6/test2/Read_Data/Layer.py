import torch
import random
import numpy as np
import pennylane as qml
from Global import *

def Read_Par():
  with open(f'../../../../2_train/L009/6_lr_-6/test2/Result/par',"r") as f:
    data=[float(num) for line in f for num in line.strip().split()]
  par=torch.tensor(data,device=dev2,dtype=Dtype).reshape(N_lay,3,N_Ugae)
  return par

@qml.qnode(dev9)
def Circ_Pen_Ful9(stat,par):
  qml.QubitStateVector(stat,wires=range(9))
  m=np.arange(9)
  for l in range (N_lay):
    q0=np.roll(m,-2*l)
    q1=np.roll(m,-2*l-1)
    for n in range (N_Ugae):
      qml.RZ(par[l][0][n],wires=m[n])
      qml.RX(par[l][1][n],wires=m[n])
      qml.RZ(par[l][2][n],wires=m[n])
    qml.Barrier(wires=range(9))
    for p in range (0,8,2):
      qml.CNOT(wires=[q0[p],q0[p+1]])
    for p in range (0,8,2):
      qml.CNOT(wires=[q1[p],q1[p+1]])
    qml.Barrier(wires=range(9))
  pro=qml.probs(wires=0)
  return pro

