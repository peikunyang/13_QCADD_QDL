import os
import numpy as np
import torch

Max=32
Div=0.5

def GridP():
  coord=[]
  for i in range (Max):
    x=(i-Max/2+0.5)*Div
    for j in range (Max):
      y=(j-Max/2+0.5)*Div
      for k in range (Max):
        z=(k-Max/2+0.5)*Div
        coord.append(x)
        coord.append(y)
        coord.append(z)
  ncoord=torch.Tensor(coord).reshape(32768,3).to("cuda")
  return ncoord

def Read_Coor(pdb,aty):
  coord=[]
  atm_r=[]
  fr=open("../../4_seperate/PDBbind_v2020/%s/%s%d"%(pdb,pdb,aty),"r")
  for line in fr:
    lx=line.split()
    coord.append(float(lx[0]))
    coord.append(float(lx[1]))
    coord.append(float(lx[2]))
    atm_r.append(float(lx[3]))
  atm_c=torch.Tensor(coord).reshape(-1,3).to("cuda")
  fr.close()
  return atm_c,torch.Tensor(atm_r).to("cuda")

def Con_Img(img,aty,atm_c,atm_r,grid):
  dis=torch.cdist(atm_c,grid,p=2)
  ratio=torch.div(dis,atm_r.reshape(-1,1))
  img0=torch.exp(torch.tensor(-2)*torch.square(ratio))
  img0_jug=torch.lt(ratio,torch.tensor(1.0))
  img0f=torch.mul(img0,img0_jug)
  img1=torch.square((torch.tensor(3.0)-torch.tensor(2.0)*ratio)/torch.exp(torch.tensor(1.0)))
  img1_jug1=torch.lt(ratio,torch.tensor(1.5))
  img1_jug2=torch.ge(ratio,torch.tensor(1.0))
  img1_jug3=torch.mul(img1_jug1,img1_jug1)
  img1f=torch.mul(img1,img1_jug3)
  img[aty]=img[aty]+(img0f+img1f).sum(0)
  return img

def Pooling1(img):
  img=img.reshape((8,16,2,16,2,16,2))
  img=torch.max(img,6)[0]
  img=torch.max(img,4)[0]
  img=torch.max(img,2)[0]
  return img

def Pooling2(img):
  img=img.reshape((8,8,2,8,2,8,2))
  img=torch.max(img,6)[0]
  img=torch.max(img,4)[0]
  img=torch.max(img,2)[0]
  return img

def Pooling3(img):
  img=img.reshape((8,4,2,4,2,4,2))
  img=torch.max(img,6)[0]
  img=torch.max(img,4)[0]
  img=torch.max(img,2)[0]
  return img

def Write_Img(pdb,img):
  fw=open("../PDBbind_v2020/%s.img"%(pdb),"w")
  img=Pooling1(img)
  img=Pooling2(img)
  img=Pooling3(img).reshape(2,-1)
  img=torch.nn.functional.normalize(img,p=2,dim=1).reshape(-1)
  img=torch.nn.functional.normalize(img,p=2,dim=0).reshape(8,-1)
  for j in range (8):
    for i in range (img.shape[1]):
      fw.write(" %12.10f"%(img[j][i]))
      if i%8==7:
        fw.write("\n")
  fw.close()

def Main():
  pdb=sorted(os.listdir("../../1_PDBbind_v2020"))
  Grid=GridP()
  for i in range (len(pdb)):
    Img=torch.zeros((8,32768)).to("cuda")
    for j in range (8):
      Atm_c,Atm_r=Read_Coor(pdb[i],j)
      Img=Con_Img(Img,j,Atm_c,Atm_r,Grid)
    Write_Img(pdb[i],Img)

Main()

