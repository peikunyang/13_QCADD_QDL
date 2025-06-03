import os
import numpy as np

def Read_Data(pdb,lig):
  atm_t=[]
  coord=[]
  atm_r=[]
  if lig==0:
    fr=open("../../3_rm_atom/PDBbind_v2020/%s/%s_rec"%(pdb,pdb),"r")
  if lig==1:
    fr=open("../../3_rm_atom/PDBbind_v2020/%s/%s_lig"%(pdb,pdb),"r")
  for line in fr:
    lx=line.split()
    atm_t.append(lx[0])
    coord.append(lx[1])
    coord.append(lx[2])
    coord.append(lx[3])
    atm_r.append(lx[4])
  atm_c=np.array(coord).reshape(-1,3)
  fr.close()
  return atm_t,atm_c,atm_r

def Write_Data(fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7,lig,atm_t,atm_c,atm_r):
  for i in range (len(atm_t)):
    if lig==0:
      if atm_t[i]=="C":
        fw0.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
      if atm_t[i]=="N":
        fw1.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
      if atm_t[i]=="O":
        fw2.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
      if atm_t[i]=="Z":
        fw3.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
    if lig==1:
      if atm_t[i]=="C":
        fw4.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
      if atm_t[i]=="N":
        fw5.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
      if atm_t[i]=="O":
        fw6.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))
      if atm_t[i]=="Z":
        fw7.write("%9s %9s %9s %3s\n"%(atm_c[i][0],atm_c[i][1],atm_c[i][2],atm_r[i]))

def MakeDir(pdb):
  for i in range (len(pdb)):
    os.mkdir("../PDBbind_v2020/%s"%(pdb[i]))

def Fopen(pdb):
  fw0=open("../PDBbind_v2020/%s/%s0"%(pdb,pdb),"w")
  fw1=open("../PDBbind_v2020/%s/%s1"%(pdb,pdb),"w")
  fw2=open("../PDBbind_v2020/%s/%s2"%(pdb,pdb),"w")
  fw3=open("../PDBbind_v2020/%s/%s3"%(pdb,pdb),"w")
  fw4=open("../PDBbind_v2020/%s/%s4"%(pdb,pdb),"w")
  fw5=open("../PDBbind_v2020/%s/%s5"%(pdb,pdb),"w")
  fw6=open("../PDBbind_v2020/%s/%s6"%(pdb,pdb),"w")
  fw7=open("../PDBbind_v2020/%s/%s7"%(pdb,pdb),"w")
  return fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7

def Fclose(fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7):
  fw0.close()
  fw1.close()
  fw2.close()
  fw3.close()
  fw4.close()
  fw5.close()
  fw6.close()
  fw7.close()
  
def Main():
  pdb=sorted(os.listdir("../../1_PDBbind_v2020"))
  MakeDir(pdb)
  for i in range (len(pdb)):
    print(pdb[i])
    fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7=Fopen(pdb[i])
    Atm_t,Atm_c,Atm_r=Read_Data(pdb[i],0)
    Write_Data(fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7,0,Atm_t,Atm_c,Atm_r)
    Atm_t,Atm_c,Atm_r=Read_Data(pdb[i],1)
    Write_Data(fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7,1,Atm_t,Atm_c,Atm_r)
    Fclose(fw0,fw1,fw2,fw3,fw4,fw5,fw6,fw7)

Main()

