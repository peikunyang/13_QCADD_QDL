import os
import numpy as np

Max=19
Min=-19

def Read_Lig(pdb):
  atm_t=[]
  coord=[]
  atm_r=[]
  fr=open("../../2_keep_atom/PDBbind_v2020/%s/%s_lig"%(pdb,pdb),"r")
  for line in fr:
    lx=line.split()
    atm_t.append(lx[0])
    coord.append(float(lx[1]))
    coord.append(float(lx[2]))
    coord.append(float(lx[3]))
    atm_r.append(lx[4])
  atm_c=np.array(coord).reshape(-1,3)
  fr.close()
  return atm_t,atm_c,atm_r

def Read_Rec(pdb):
  atm_t=[]
  coord=[]
  atm_r=[]
  fr=open("../../2_keep_atom/PDBbind_v2020/%s/%s_rec"%(pdb,pdb),"r")
  for line in fr:
    lx=line.split()
    atm_t.append(lx[0])
    coord.append(float(lx[1]))
    coord.append(float(lx[2]))
    coord.append(float(lx[3]))
    atm_r.append(lx[4])
  atm_c=np.array(coord).reshape(-1,3)
  fr.close()
  return atm_t,atm_c,atm_r

def Write_Lig(pdb,atm_t,atm_c,atm_r):
  cen=np.mean(atm_c,axis=0)
  natm_c=atm_c-cen
  fw=open("../PDBbind_v2020/%s/%s_lig"%(pdb,pdb),"w")
  for i in range (len(atm_t)):
    if ((natm_c[i][0]<Max)&(natm_c[i][0]>Min)&(natm_c[i][1]<Max)&(natm_c[i][1]>Min)&(natm_c[i][2]<Max)&(natm_c[i][2]>Min)):
      fw.write("%1s %9.4f %9.4f %9.4f %3s\n"%(atm_t[i],natm_c[i][0],natm_c[i][1],natm_c[i][2],atm_r[i]))
  fw.close()
  return cen

def Write_Rec(pdb,atm_t,atm_c,atm_r,cen):
  fw=open("../PDBbind_v2020/%s/%s_rec"%(pdb,pdb),"w")
  natm_c=atm_c-cen
  for i in range (len(atm_t)):
    if ((natm_c[i][0]<Max)&(natm_c[i][0]>Min)&(natm_c[i][1]<Max)&(natm_c[i][1]>Min)&(natm_c[i][2]<Max)&(natm_c[i][2]>Min)):
      fw.write("%1s %9.4f %9.4f %9.4f %3s\n"%(atm_t[i],natm_c[i][0],natm_c[i][1],natm_c[i][2],atm_r[i]))
  fw.close()

def MakeDir(pdb):
  for i in range (len(pdb)):
    os.mkdir("../PDBbind_v2020/%s"%(pdb[i]))

def Main():
  pdb=sorted(os.listdir("../../1_PDBbind_v2020"))
  MakeDir(pdb)
  for i in range (len(pdb)):
    print(pdb[i])
    Atm_t,Atm_c,Atm_r=Read_Lig(pdb[i])
    Cen=Write_Lig(pdb[i],Atm_t,Atm_c,Atm_r)
    Atm_t,Atm_c,Atm_r=Read_Rec(pdb[i])
    Write_Rec(pdb[i],Atm_t,Atm_c,Atm_r,Cen)

Main()

