import numpy as np
import torch
import pennylane as qml

N_lay=10
N_Ugae=9
N_Ite=2000
learning_rate=1E-6
Batch=100

Dtype=torch.float
Dtypec=torch.cfloat
dev2="cuda"

dev9=qml.device("default.qubit",wires=9)

con0=torch.complex(torch.tensor([0],dtype=Dtype),torch.tensor([0],dtype=Dtype)).to(dev2)
con1=torch.complex(torch.tensor([1],dtype=Dtype),torch.tensor([0],dtype=Dtype)).to(dev2)
coni=torch.complex(torch.tensor([0],dtype=Dtype),torch.tensor([1],dtype=Dtype)).to(dev2)

def Normalize(stat,d):
  stat_n=torch.nn.functional.normalize(stat,p=2.0,dim=d)
  return stat_n

def Prob(img):
  bit=np.zeros((9),dtype=bool)
  count=np.zeros((512),dtype=int).reshape(2,2,2,2,2,2,2,2,2)
  for states in img:
    b0=int(states[0])
    b1=int(states[1])
    b2=int(states[2])
    b3=int(states[3])
    b4=int(states[4])
    b5=int(states[5])
    b6=int(states[6])
    b7=int(states[7])
    b8=int(states[8])
    count[b0,b1,b2,b3,b4,b5,b6,b7,b8]=img[states]
  count=count.reshape(-1)
  pro=(count/N_shot9).reshape(8,-1)[:,0].sum(0)
  return pro

