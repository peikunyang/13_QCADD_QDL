import sys
import numpy as np
import torch
import math
from Read_Data.Layer import *
from Global import *

def Read_PDB():
  pdb_gen=[]
  pdb_ref=[]
  bfe_gen=[]
  bfe_ref=[]
  fr=open("training_data/pkd_all","r")
  for line in fr:
    lx=line.split()
    pdb_gen.append(lx[0])
    bfe_gen.append(float(lx[2]))
  fr.close()
  fr=open("training_data/pkd_Coreset","r")
  for line in fr:
    lx=line.split()
    pdb_ref.append(lx[0])
    bfe_ref.append(float(lx[2]))
  fr.close()
  bfe2_gen=torch.FloatTensor(bfe_gen).to(dev2)
  bfe2_ref=torch.FloatTensor(bfe_ref).to(dev2)
  return pdb_gen,pdb_ref,bfe2_gen,bfe2_ref

def Read_Img(pdb):
  img=[]
  for i in range (len(pdb)):
    fr=open("../../../../../1_database/1_structure/5_img/1_9Q/PDBbind_v2020/%s.img"%(pdb[i]),"r")
    for line in fr:
      lx=line.split()
      for j in range (len(lx)):
        img.append(float(lx[j]))
  img2=torch.tensor(img,dtype=Dtype).reshape(len(pdb),512).to(dev2)
  img3=Normalize(img2,1)
  return img3

def Out_Energy(fn,bfe,e_pen):
  fw=open("Result/%s"%(fn),"w")
  for i in range (len(bfe)):
    fw.write("%7.2f %7.2f\n"%(bfe[i],e_pen[i]))
  fw.close()

def Out_E_diff(fw,i,bfe_gen,e_gen_pyt):
  squared_diff = [(x - y) ** 2 for x, y in zip(bfe_gen, e_gen_pyt)]
  rmsd = math.sqrt(sum(squared_diff) / len(bfe_gen))
  fw.write("%4d %7.2f\n"%(i,rmsd))
  fw.flush()

