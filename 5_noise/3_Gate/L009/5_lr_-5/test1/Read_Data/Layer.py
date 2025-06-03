import torch
import random
import numpy as np
import pennylane as qml
from Global import *

def Read_Par():
  with open(f'../../../../../2_train/L009/5_lr_-5/test1/Result/par',"r") as f:
    data=[float(num) for line in f for num in line.strip().split()]
  par=torch.tensor(data,device=dev2,dtype=Dtype).reshape(N_lay,3,N_Ugae)
  return par

@qml.qnode(dev9)
def Circ_Pen_Ful9(stat, par):
    qml.QubitStateVector(stat, wires=range(9))
    m = np.arange(9)
    def noisy_gate(gate_fn, theta, wire):
        gate_fn(theta, wires=wire)
        qml.AmplitudeDamping(0.001, wires=wire)
        qml.DepolarizingChannel(0.0005, wires=wire)
    def noisy_cnot(control, target):
        qml.CNOT(wires=[control, target])
        qml.AmplitudeDamping(0.001, wires=control)
        qml.DepolarizingChannel(0.0005, wires=control)
        qml.AmplitudeDamping(0.001, wires=target)
        qml.DepolarizingChannel(0.0005, wires=target)
    for l in range(N_lay):
        q0 = np.roll(m, -2 * l)
        q1 = np.roll(m, -2 * l - 1)
        for n in range(N_Ugae):
            noisy_gate(qml.RZ, par[l][0][n], m[n])
            noisy_gate(qml.RX, par[l][1][n], m[n])
            noisy_gate(qml.RZ, par[l][2][n], m[n])
        qml.Barrier(wires=range(9))
        for p in range(0, 8, 2):
            noisy_cnot(q0[p], q0[p + 1])
        for p in range(0, 8, 2):
            noisy_cnot(q1[p], q1[p + 1])
        qml.Barrier(wires=range(9))
    return qml.probs(wires=0)

