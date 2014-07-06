# -*- coding: utf-8 -*-
"""
Example of the ptychograpic reconstruction of simulated data. 

@author: ondrej
"""
from pylab import *
import ptycho

##################
# Simulaiton of the ptychography data: 
obj_info = {'type':'phase_ampl', 'phase_stretch':pi/5, 'alpha_win':.2} # or one can pass the obj (2D complex image) into the Simulation: s = ptycho.Simulation(obj = img) ....
probe_info = {'type':'gauss','sigma_pix': (20,40),'shape': (256,256)} # 'FZP', or 'flat' are other possibilities
scan_info = {'type': 'spiral', 'scan_step_pix': 30, 'n_scans': 50} #50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns 
data_info = {'pix_size_direct_nm': 10, 'num_phot_max': 1e6, 'bg': 0}

# Initialisation of the simulation with specivied parameters, specific <object>, <probe> or <scan>positions can be passed as s = ptycho.Simulation(obj=<object>, probe=<probe>, scan = <scan>) omitting obj_info, probe_info or scan_info (or passing it as empty dictionary "{}")
s = ptycho.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info, data_info=data_info)

# Data simulation: probe.show(), obj.show(), scan.show() and s.show_illumination() will visualise (if plottools module is available). 
s.make_data()

pos = s.scan.values
ampl = s.amplitude.values # square root of the measured diffraction pattern intensity 

##################
# Evaluation: 
# Size of the reconstructed object (obj)
nx,ny = ptycho.objShape(pos, ampl.shape[1:])

# Evaluation method
method = 'Thibault2009'
# or one can pass the method as a dictionry along with some parameteres of the update:
#method = {'method': 'Thibault2009', 'reg_const_object':0.01,'reg_const_probe':0.01} 
#method = 'Maiden2009'
#method={'method': 'Maiden2009','learning_const_object':1,'learning_const_probe':1}
savethis = True
for evaluation in (1,):
    print "\nEvaluation: %g" %evaluation            

    # Initial obj
    obj_init_info = {'type':'flat','shape':(nx,ny)}
    # Initial porobe
    probe_init_info = {'type':'gauss','shape': s.probe.values.shape, 'sigma_pix':(30,50)}    
    init = ptycho.Simulation(obj_info=obj_init_info, probe_info=probe_init_info)
    init.make_obj()
    init.make_probe()
    
    # Initialisation of the data object p with ptychograhic data (views), initial probe (probe0.values) and initial object (obj0.values)
    p = ptycho.Ptycho2D(amplitudes=ampl, positions=pos, probe0=init.probe.values, obj0=init.obj.values)
    
    # first update only object and keep the probe
    p.niterUpProbeFalse = 50
    p.Run(p.niterUpProbeFalse,verbose=10,updateProbe=False,method=method)

    # updating both object and probe
    p.niterUpProbeTrue = 400
    p.Run(p.niterUpProbeTrue,verbose=10,updateProbe=True,method=method)
    
    # Saving results (initial probe and object, reconstructed probe and object and convergence curve (R))
    if savethis:
        resdir='Results'
        #appApp = ('_scanINT_R%.4f' % p.R()).replace('.', '-') #'_R'+str(int(p.R()*10000))          
        appApp =''
        nameApp='_niterUpdateObj'+str(p.niterUpProbeFalse)+'_niterUpdateObjProbe'+str(p.niterUpProbeTrue)+'_method'+p.method+'_eval'+ str(evaluation)+appApp
        p.SaveResults(resdir=resdir, name_appendix = nameApp)