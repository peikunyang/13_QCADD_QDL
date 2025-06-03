import torch
import random
import numpy as np
import pennylane as qml
from Global import *

def Para():
  par=torch.rand((N_lay,3,N_Ugae),device=dev2,dtype=Dtype,requires_grad=True)
  return par

def Read_Para():
  par=[]
  fr=open("par_run1","r")
  p=0
  for line in fr:
    if p==0:
      p=p+1
    else:
      lx=line.split()
      for i in range (len(lx)):
        par.append(float(lx[i]))
  par=torch.tensor(par,device=dev2,dtype=Dtype).reshape(N_lay,3,N_Ugae)
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

def Identity():
  c00=con1
  c01=con0
  c10=con0
  c11=con1
  c0=torch.stack([c00,c01],0)
  c1=torch.stack([c10,c11],0)
  c2=torch.stack([c0,c1],0)
  return torch.squeeze(c2)

def Hadamard():
  c00=con1/torch.sqrt(torch.tensor(2)).to(dev2)
  c01=con1/torch.sqrt(torch.tensor(2)).to(dev2)
  c10=con1/torch.sqrt(torch.tensor(2)).to(dev2)
  c11=-1*con1/torch.sqrt(torch.tensor(2)).to(dev2)
  c0=torch.stack([c00,c01],0)
  c1=torch.stack([c10,c11],0)
  c2=torch.stack([c0,c1],0)
  return torch.squeeze(c2)

def RX(thit):
  c00=con1*torch.cos(thit/2)
  c01=-1*coni*torch.sin(thit/2)
  c10=-1*coni*torch.sin(thit/2)
  c11=con1*torch.cos(thit/2)
  c0=torch.stack([c00,c01],0)
  c1=torch.stack([c10,c11],0)
  c2=torch.stack([c0,c1],0)
  return torch.squeeze(c2)

def RY(thit):
  c00=con1*torch.cos(thit/2)
  c01=-1*con1*torch.sin(thit/2)
  c10=con1*torch.sin(thit/2)
  c11=con1*torch.cos(thit/2)
  c0=torch.stack([c00,c01],0)
  c1=torch.stack([c10,c11],0)
  c2=torch.stack([c0,c1],0)
  return torch.squeeze(c2)

def RZ(thit):
  c00=torch.exp(-1*coni*thit/2)
  c01=con0
  c10=con0
  c11=torch.exp(coni*thit/2)
  c0=torch.stack([c00,c01],0)
  c1=torch.stack([c10,c11],0)
  c2=torch.stack([c0,c1],0)
  return torch.squeeze(c2)

def CNOT():
  c00=con1
  c01=con0
  c02=con0
  c03=con0
  c10=con0
  c11=con1
  c12=con0
  c13=con0
  c20=con0
  c21=con0
  c22=con0
  c23=con1
  c30=con0
  c31=con0
  c32=con1
  c33=con0
  d0=torch.stack([c00,c01,c02,c03],0)
  d1=torch.stack([c10,c11,c12,c13],0)
  d2=torch.stack([c20,c21,c22,c23],0)
  d3=torch.stack([c30,c31,c32,c33],0)
  e=torch.stack([d0,d1,d2,d3],0)
  return torch.squeeze(e)

def FCNOT9():
  c0=CNOT()
  c1=CNOT()
  c2=CNOT()
  c3=CNOT()
  c4=Identity()
  d0=torch.kron(c3,c4)
  d1=torch.kron(c2,d0)
  d2=torch.kron(c1,d1)
  d3=torch.kron(c0,d2)
  return torch.squeeze(d3)

def Full_Con9(Ugate,par):
  c0=Ugate(par[0])
  c1=Ugate(par[1])
  c2=Ugate(par[2])
  c3=Ugate(par[3])
  c4=Ugate(par[4])
  c5=Ugate(par[5])
  c6=Ugate(par[6])
  c7=Ugate(par[7])
  c8=Ugate(par[8])
  d0=torch.kron(c7,c8)
  d1=torch.kron(c6,d0)
  d2=torch.kron(c5,d1)
  d3=torch.kron(c4,d2)
  d4=torch.kron(c3,d3)
  d5=torch.kron(c2,d4)
  d6=torch.kron(c1,d5)
  d7=torch.kron(c0,d6)
  return torch.squeeze(d7)

def Circ_Tor_Ful9(img,par,batch):
  img=img.reshape(batch,-1).permute(1,0)
  p=np.arange(9)
  for l in range (N_lay):
    img=torch.matmul(Full_Con9(RZ,par[l][0]),img)
    img=torch.matmul(Full_Con9(RX,par[l][1]),img)
    img=torch.matmul(Full_Con9(RZ,par[l][2]),img)
    q0=np.roll(p,-2*l)
    q1=np.roll(p,-1)
    q2=np.roll(p,2*l+1)
    img=img.reshape(2,2,2,2,2,2,2,2,2,batch).permute(q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],9).reshape(512,batch)
    img=torch.matmul(FCNOT9(),img).reshape(2,2,2,2,2,2,2,2,2,batch).permute(q1[0],q1[1],q1[2],q1[3],q1[4],q1[5],q1[6],q1[7],q1[8],9).reshape(512,batch)
    img=torch.matmul(FCNOT9(),img).reshape(2,2,2,2,2,2,2,2,2,batch).permute(q2[0],q2[1],q2[2],q2[3],q2[4],q2[5],q2[6],q2[7],q2[8],9).reshape(512,batch)
  img=img.permute(1,0)
  pro=torch.mul(img,torch.conj(img)).real.float()
  pro=pro.reshape(batch,2,256).sum(2)
  return pro

