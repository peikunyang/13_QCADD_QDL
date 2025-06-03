import os
import numpy as np

def Radius(atm_t):
  if atm_t=="Z":
    r=2.0
  if atm_t=="C":
    r=1.9
  if atm_t=="N":
    r=1.8
  if atm_t=="O":
    r=1.7
  return r

def Read_Lig(pdb):
  atm_t=[]
  coord=[]
  n=0
  fr=open("../../1_PDBbind_v2020/%s/%s_ligand.sdf"%(pdb,pdb),"r")
  for line in fr:
    lx=line.split()
    if ((n>=4)&(len(lx)>=9)):
      if (lx[3]!="H"):
        if ((lx[3]!="N")&(lx[3]!="C")&(lx[3]!="O")):
          lx[3]="Z"
        atm_t.append(lx[3])
        coord.append(lx[0])
        coord.append(lx[1])
        coord.append(lx[2])
    n=n+1
  atm_c=np.array(coord).reshape(-1,3)
  fr.close()
  return atm_t,atm_c

def Read_Rec(pdb):
  atm_t=[]
  coord=[]
  fr=open("../../1_PDBbind_v2020/%s/%s_protein.pdb"%(pdb,pdb),"r")
  for line in fr:
    if ((line[:4]=="ATOM")|(line[:6]=="HETATM")):
      at=line[13]
      if (at!="H"):
        if ((at!="N")&(at!="C")&(at!="O")):
          at="Z"
        atm_t.append(at)
        coord.append(line[30:38])
        coord.append(line[38:46])
        coord.append(line[46:54])
  atm_c=np.array(coord).reshape(-1,3)
  fr.close()
  return atm_t,atm_c

def Write_Lig(pdb,atm_t,atm_c):
  fw=open("../PDBbind_v2020/%s/%s_lig"%(pdb,pdb),"w")
  for i in range (len(atm_t)):
    fw.write("%1s %9s %9s %9s %3.1f\n"%(atm_t[i],atm_c[i][0],atm_c[i][1],atm_c[i][2],Radius(atm_t[i])))
  fw.close()

def Write_Rec(pdb,atm_t,atm_c):
  fw=open("../PDBbind_v2020/%s/%s_rec"%(pdb,pdb),"w")
  for i in range (len(atm_t)):
    fw.write("%1s %9s %9s %9s %3.1f\n"%(atm_t[i],atm_c[i][0],atm_c[i][1],atm_c[i][2],Radius(atm_t[i])))
  fw.close()

def MakeDir(pdb):
  for i in range (len(pdb)):
    os.mkdir("../PDBbind_v2020/%s"%(pdb[i]))

def Main():
  pdb=sorted(os.listdir("../../1_PDBbind_v2020"))
  MakeDir(pdb)
  for i in range (len(pdb)):
    Atm_t,Atm_c=Read_Lig(pdb[i])
    Write_Lig(pdb[i],Atm_t,Atm_c)
    Atm_t,Atm_c=Read_Rec(pdb[i])
    Write_Rec(pdb[i],Atm_t,Atm_c)

Main()

