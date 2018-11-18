import numpy as np
import scipy
import matplotlib
import pylab
from scipy.integrate import odeint

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


InitialCond=[0.75,  #ArAr3   0
             0.25,  #ArAr1   1
             0.,  #ArXe3   2
             0.,  #ArXe1   3
             0.,  #XeXe3   4
             0.,  #XeXe1   5
             0.,  # 128    6
             0.,  # 145    7
             0.   # 175    8
            ]

Gams       =[-1./1450,#ArAr3
            -1./6.,  #ArAr1
            -1./800, #ArXe3
            -1./5,   #ArXe1
            -1./45., #XeXe3
            -1./3.,  #XeXe1
            0,       # 128
            0,       # 145
            0        # 175
            ]

Labels = ["ArAr3",
          "ArAr1",
          "ArXe3",
          "ArXe1",
          "XeXe3",
          "XeXe1",
          "128 nm",
          "145 nm",
          "175 nm",
]
Elts=len(Gams)
CollisionMatrix=np.ndarray(shape=(Elts,Elts,Elts))
CollisionMatrix=np.zeros_like(CollisionMatrix)

XeConc=2e-5
norm=100
k_ArAr3_Xe_ArXe = 100
k_ArAr1_Xe_ArXe = 100
k_ArXe_Xe_XeXe1 = 500
k_ArXe_Xe_XeXe3 = 500


def SimpleModelDT(inp,t):
    outp=np.zeros_like(inp)
    outp[0] = (Gams[0] - k_ArAr3_Xe_ArXe*XeConc)*inp[0]
    outp[1] = (Gams[1] - k_ArAr1_Xe_ArXe*XeConc)*inp[1]
    outp[2] = (Gams[2] -(k_ArXe_Xe_XeXe1+k_ArXe_Xe_XeXe3)*XeConc)*inp[2]+k_ArAr3_Xe_ArXe*XeConc*inp[0]+k_ArAr1_Xe_ArXe*XeConc*inp[1]
    outp[3] = 0
    outp[4] = Gams[4]*inp[4]+k_ArXe_Xe_XeXe1*XeConc*inp[2]
    outp[5] = Gams[5]*inp[5]+k_ArXe_Xe_XeXe3*XeConc*inp[2]
    outp[6] = - (Gams[0]*inp[0]+Gams[1]*inp[1])
    outp[7] = - (Gams[2]*inp[2]+Gams[3]*inp[3])
    outp[8] = - (Gams[4]*inp[4]+Gams[5]*inp[5])
    return outp



def SolveAndPlot(Log_XeConc,k_ArAr3_Xe_ArXe=100,k_ArAr1_Xe_ArXe = 100,k_ArXe_Xe_XeXe1 = 500,k_ArXe_Xe_XeXe3 = 500,Ar3NonRadiative=3000):
    TheXeConc=pow(10,Log_XeConc)*1e-6
    t = np.linspace(0,1500,200)
    def TunedModelDT(inp,t):
        outp=np.zeros_like(inp)
        outp[0] = (Gams[0] - k_ArAr3_Xe_ArXe*TheXeConc-1./Ar3NonRadiative)*inp[0]
        outp[1] = (Gams[1] - k_ArAr1_Xe_ArXe*TheXeConc)*inp[1]
        outp[2] = (Gams[2] -(k_ArXe_Xe_XeXe1+k_ArXe_Xe_XeXe3)*TheXeConc)*inp[2]+k_ArAr3_Xe_ArXe*TheXeConc*inp[0]+k_ArAr1_Xe_ArXe*XeConc*inp[1]
        outp[3] = 0
        outp[4] = Gams[4]*inp[4]+k_ArXe_Xe_XeXe1*TheXeConc*inp[2]
        outp[5] = Gams[5]*inp[5]+k_ArXe_Xe_XeXe3*TheXeConc*inp[2]
        outp[6] = - (Gams[0]*inp[0]+Gams[1]*inp[1])
        outp[7] = - (Gams[2]*inp[2]+Gams[3]*inp[3])
        outp[8] = - (Gams[4]*inp[4]+Gams[5]*inp[5])
        return outp
    
    # solve ODE
    y = odeint(TunedModelDT,InitialCond,t)
    fig, [axs1,axs2,axs3,axs4]=pylab.subplots(1,4,figsize=(15,4))

    for i in range(0,len(y[0])-3):
        axs1.plot(t,y[:,i],label=Labels[i])
    axs1.set_title("Excimer populations")

    axs1.legend(loc='upper right')
    axs1.set_xlabel("Time / ns")

    Emis128=(y[1:,6]-y[:-1,6])/(t[1]-t[0])
    Emis145=(y[1:,7]-y[:-1,7])/(t[1]-t[0])
    Emis175=(y[1:,8]-y[:-1,8])/(t[1]-t[0])

    NormMax=np.max(Emis128+Emis145+Emis175)
    Emis128/=NormMax
    Emis145/=NormMax
    Emis175/=NormMax    
    
    axs2.plot(t[0:-1],Emis128,color='DarkBlue',label='128 nm')
    axs2.plot(t[0:-1],Emis145,color='purple',label='145 nm')
    axs2.plot(t[0:-1],Emis175,color='DarkRed',label='175 nm')
    axs2.legend(loc='center right')
    axs2.set_xlabel("Time / ns")
    axs2.set_ylim(0,0.1)
    axs2.set_title("Light Yields")
    axs3.bar(['128','145','175','Total'],[sum(Emis128),sum(Emis145),sum(Emis175),sum(Emis128+Emis175+Emis145)],color=['DarkBlue','purple','DarkRed','black'])
    axs3.set_title("Total Light")
    axs2.set_xlabel("Wavelength")

    axs4.set_xlim(0,10)
    axs4.set_ylim(0.5,7)
    axs4.set_xticks([])
    axs4.set_yticks([])
    axs4.text(1,6,'Xe Concentration  \n  {0:1.1f} ppm'.format(TheXeConc*1e6) )
    axs4.text(1,5,'k_ArAr3_Xe_ArXe   \n  {0:1.1f} (arb)'.format(k_ArAr3_Xe_ArXe) )
    axs4.text(1,4,'k_ArAr1_Xe_ArXe   \n  {0:1.1f} (arb)'.format(k_ArAr1_Xe_ArXe) )
    axs4.text(1,3,'k_ArXe_Xe_XeXe1   \n  {0:1.1f} (arb)'.format(k_ArXe_Xe_XeXe1) )
    axs4.text(1,2,'k_ArXe_Xe_XeXe3   \n  {0:1.1f} (arb)'.format(k_ArXe_Xe_XeXe3) )
    axs4.text(1,1,'ArAr3 Non-rad t   \n  {0:1.1f} ns'.format(Ar3NonRadiative) )
#    axs3.set_xticks([1,2,3,4],['128','145','175','Total'])
#    axs2.semilogy()
