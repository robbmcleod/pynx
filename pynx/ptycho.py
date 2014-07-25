########################################################################
#
# Ptychographic Reconstruction
#
#  Vincent Favre-Nicolin@cea.fr  CEA Grenoble / INAC / SP2M
#
########################################################################
from __future__ import division
import sys, os, time, glob
import numpy as np

from scipy import log10, sqrt, pi, exp
from scipy.fftpack import ifftshift, fftshift, fft2, ifft2
from scipy.optimize import minimize

from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.cm as cm

################################################
################################################
#
# Auxiliary functions
#
################################################
################################################


def plotR(filename = "Results/R_*",showLegend = True, showNumbers = True, plotGraph = True, fontsize = 5):
    """
    Plots convergence curves.
    
    Examples:
    plotR() - plots all convergence curves stored under default name Results/R_*
    plotR(filename=<name_file_pattern>) plots all curves matching <name_file_pattern>: e.g. plotR(filename='Results/R_*Thibaut*') - plots all curves containing "R_" and "Thibault"
    set showLegend=False not to show the legend.
    
    Filename of the best object and probe:
    r = ptycho.plotR()
    obj_best = r[0]%'obj' 
    probe_best = r[0]%'probe'
    """
    plt.figure()
    leg = []
    Rmin=1.
    index = 0
    files = glob.glob(filename)
    rall = np.zeros(len(files))
    nfiles = len(files)
    colors = cm.rainbow(np.linspace(0, 1, nfiles))
    for filename in files:
        R = np.load(filename)
        rall[index] = R[-1]        
        index += 1  
    
    ind_sort = [i[0] for i in sorted(enumerate(rall), key=lambda x:x[1])]
    files_sort =  list( files[i] for i in ind_sort)
    rall_sort =  list( rall[i] for i in ind_sort)
    index = 0
    for filename, c in zip(files_sort, colors):
        R = np.load(filename)
        plt.semilogy(R,c=c)
        if showNumbers: 
            plt.text(len(R),R[-1],str(index))
        leg.append(str(index) + ' ' + filename)
        index += 1

    plt.grid(b=True, which = 'both')    
    if showLegend:
        plt.legend(leg, fontsize = fontsize)
        
    evMin = files_sort[0]    
    try:
        evMin = evMin.replace("R_","%s_") # one can load e.g. best object: by load(evMin%'obj')
    except:
        pass
    Rmin = rall_sort[0]
    print "Best: " + evMin + " @ " + str(Rmin)
    return evMin, rall_sort, leg 
    
def get_res(filename = "Results/obj_*"):
    """
    Returns all images matching given pattern.

    Example: 
    imgs = getIms(filename = "Results/obj_*")
    imgs[0] - array containing all saved reconsructe objects
    imgs = getIms(filename = "Results/probe_*Thibault*")
    imgs[0] - array containing all probes containing "Thibault" in their names
    
    """
    files = glob.glob(filename)
    im_all = np.load(files[0])[np.newaxis]
    for filename in files[1:]:
        im = np.load(filename)
        im_all = np.concatenate((im_all, im[np.newaxis]),0)        
    return im_all, files

def plot_scan(posVert,posHoriz,val=1,xlab=None, ylab=None, vlab=None, show_num=True):
    """
    Plots positions of the scan.
    posVert, posHoriz - vertical and horizontal positions
    val - optional - one can pas e.g. val=(amplitudes**2).sum(2).sum(1) which will show each marker coloured according to integrated intesnisty
    xlab,ylab - optional - labels of the axis
    vlab - optional - label for the colorbar
    show_num - if set to True, the numbers at each point are shown    
    """
    plt.figure(); plt.scatter(posHoriz,posVert,c=val,s=40) # notation in Ptycho is first coordinate vertical and second horizontal... 
    if show_num:
        plt.plot(posHoriz,posVert,':')
        for i in range(len(posVert)):
            plt.annotate(str(i), xy = (posHoriz[i], posVert[i]), xytext = (5, 0), textcoords = 'offset points')
    plt.gca().invert_yaxis()    
    plt.axis('equal')
    plt.grid(b=True, which='both')    
    if not isinstance(val,int):
        cbr = plt.colorbar(format='%.1e')
        cbr.set_label(vlab, rotation=90)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)

def phase2rgb(s):
   """
   Crates RGB image with colour-coded phase. 
   """
   ph=np.angle(s)
   t=pi/3
   nx,ny=s.shape   
   rgba=np.zeros((nx,ny,4))   
   rgba[:,:,0]=(ph<t)*(ph>-t) + (ph>t)*(ph<2*t)*(2*t-ph)/t + (ph>-2*t)*(ph<-t)*(ph+2*t)/t
   rgba[:,:,1]=(ph>t)         + (ph<-2*t)      *(-2*t-ph)/t+ (ph>0)*(ph<t)    *ph/t
   rgba[:,:,2]=(ph<-t)        + (ph>-t)*(ph<0) *(-ph)/t + (ph>2*t)         *(ph-2*t)/t
   return rgba

def complex2rgbalog(s,amin=0.5,dlogs=2):
   """
   Returns RGB image with colour-coded phases and log10(amplitude) in birghtness. 
   """
   rgba = phase2rgb(s)
   a = np.log10(abs(s)+1e-20)
   a -= a.max()-dlogs # display dlogs orders of magnitude
   rgba[:,:,3] = amin+a/dlogs*(1-amin)*(a>0)
   return rgba

def complex2rgbalin(s):
   """
   Returns RGB image with with colour-coded phase and log10(amplitude) in birghtness. 
   """
   rgba = phase2rgb(s)
   a = np.abs(s)
   a /= a.max()
   rgba[:,:,3] = a
   return rgba

def objShape(pos, probe_shape):
    """
    Determines the required size for the reconstructed object. 
    """
    nx = 2*(abs(np.ceil(pos[0]))).max() + probe_shape[0]
    ny = 2*(abs(np.ceil(pos[1]))).max() + probe_shape[1]    

    return nx,ny
    
def showProgress(obj,probe,tit1='Object',tit2='Probe',stit=None, figNum=100):
      """
      Plots the progress of the ptychographic evaluation. 
      """
      gs = gridspec.GridSpec(2, 2, height_ratios=[1, probe.shape[0]/obj.shape[0]]) 
      plt.ion()
      fig = plt.figure(figNum)
      
      ax0 = plt.subplot(gs[0])
      ax0.imshow(np.abs(obj))
      plt.title(tit1 + ' modulus')      

      ax1 = plt.subplot(gs[1])
      ax1.imshow(np.angle(obj))      
      plt.title(tit1 + ' phase')            

      ax2 = plt.subplot(gs[2])
      ax2.imshow(np.abs(probe))           
      plt.title(tit2 + ' modulus')            
      
      ax2 = plt.subplot(gs[3])      
      ax2.imshow(np.angle(probe))                           
      plt.title(tit2 + ' phase')            
      
      if stit:
          txt = fig.suptitle('')
          txt.set_text(stit)
      plt.draw()
      

def showFT(p):
      a=None
      for d in p.views:
          if a==None:
              a=d.s_calc0[np.newaxis,:]
          else:
              a=np.concatenate((a,d.s_calc0[np.newaxis,:]),axis=0)          
      #plt.close(101)
      #plotting.imTiles(np.log(abs(a)))
      plt.figure()
      plotting.im(np.log(abs(a[0])))
      plt.draw()        

        
def get_pos(p,show_plot=False):
    rx = np.zeros(len(p.views))
    ry = np.zeros(len(p.views))    
    for i,v in enumerate(p.views):
        rx[i],ry[i] = v.dx, v.dy
    if show_plot:
        plt.figure();
        plt.plot(rx,ry,marker='o',color='r')
        plt.axis('equal')
        plt.grid(b=True, which='both')    
    return (rx,ry)

def get_s_calc0(p):
    sx, sy = p.views[0].s_calc0.shape
    sz = len(p.views)
    s_calc0 = np.zeros( (sz,sx,sy) ,dtype=p.views[0].s_calc0.dtype)
    for i,v in enumerate(p.views):
        s_calc0[i] = v.s_calc0
    return s_calc0

def get_view_coord(obj_shape, probe_shape, shift):
    cx = (obj_shape[0]-probe_shape[0])//2 + shift[0]
    cy = (obj_shape[1]-probe_shape[1])//2 + shift[1]
    return cx, cy
    
def gradPhase(p,im,mask):
    x = np.linspace(-im.shape[0]/2,(im.shape[0]/2-1),im.shape[0]).astype(np.float32)
    yy,xx = np.meshgrid(x,x)
    oPhaseMasked = np.ma.masked_array(np.angle(im*np.exp(1j*2*np.pi*(xx*p[0]/im.shape[0]+yy*p[1]/im.shape[1]))),-mask)
    dx,dy = np.gradient(oPhaseMasked)
#    return sum(abs(dx[mask])+abs(dy[mask]))
    return np.median(abs(dx[mask])+abs(dy[mask]))

def minimizeGradPhase(im, mask_thr = 0.3, init_grad = [0,0]):
    """
    Minimises the gradient in the phase of the input image im. 
    """
    mask = (abs(im)/abs(im).max()) > mask_thr
    res = minimize(gradPhase,init_grad, args=(im,mask,), method='Powell',options={'xtol': 1e-18, 'disp': True})
    x = np.linspace(-im.shape[0]/2,(im.shape[0]/2-1),im.shape[0]).astype(np.float32)
    yy,xx = np.meshgrid(x,x)
    gradCorr=np.exp(1j*2*np.pi*(xx*res['x'][0]/im.shape[0]+yy*res['x'][1]/im.shape[1]))    
    print res['x']
    return im*gradCorr, gradCorr, mask, res['x']
    

################################################
################################################
#
# Ptychographic iterative reconstruction
#
################################################
################################################


def MakePtychoData(amplitudes,dx,dy):
    """
    Creates ptychographic dataset from the stack of amplitudes and corresponding positions.
    s_obs: amplitudes (i.e. sqrt(measured intensity of the diffraction))
    dx: vertical position
    dy: horizontal position
    """
    views=[]
    for i in np.arange(amplitudes.shape[0]):
       posx = dx[i]
       posy = dy[i]
       frame = amplitudes[i]
       views.append(PtychoDiffData2D(frame,posx,posy)) #o the shift posx,posy are in pixels
    return views
    
class PtychoDiffData2D:
  """
  Creates the ptychographic set from the amplitudes and corresponding positions of the scan. 
  s_obs are the amplitudes (i.e. sqrt(measured intensity of the diffraction))!
  """
  def __init__(self,s_obs,dx,dy):
    self.dx = dx # this is an amount of the shift between the different views
    self.dy = dy
    self.s_obs = s_obs #o observed data

def ePIEupdate(A,psi,psi0,const):
      "Maiden2009 update for either probe or object."      
      return const*A.conjugate()*(psi-psi0)/(abs(A)**2).max()

class Ptycho2D:
  """
  Reconstruction of the ptychograhic data.
  """
  def __init__(self, amplitudes, positions, probe0, obj0, prec='single'):
    if prec is 'single':    # precision of the computation
        self.DTYPE_REAL = np.float32    #dtype of the real arrays
        self.DTYPE_CPLX = np.complex64    #dtype of the complex arrays
    elif prec is 'double':
        self.DTYPE_REAL = np.float64
        self.DTYPE_CPLX = np.complex128
    print 'Using %s precision.'%prec

    self.views = MakePtychoData(self.DTYPE_REAL(amplitudes), positions[0], positions[1])
    self.nbproc = os.sysconf("SC_NPROCESSORS_ONLN")
    self.probe = probe0.astype(self.DTYPE_CPLX)
    self.probe0 = probe0.copy()
    self.obj = obj0.astype(self.DTYPE_CPLX)
    self.obj0 = obj0.copy()

    try: # to do: revise this to use fftshift(fft2(fftshift(A)))
      # use FFTW
      if self.DTYPE_CPLX is np.complex128: # double precision
          import fftw3 as fftw
      elif self.DTYPE_CPLX is np.complex64: # single precision
          import fftw3f as fftw
          
      self.fftw_data = np.zeros(self.views[0].s_obs.shape, dtype=self.DTYPE_CPLX)
      # create a forward and backward fft plan
      self.plan_fft = fftw.Plan(self.fftw_data,None, direction='forward', flags=['measure'],nthreads=4)
      self.plan_ifft = fftw.Plan(self.fftw_data, None, direction='backward', flags=['measure'],nthreads=4)
      self.useFFTW3 = True
      #o tmp=self.fft(self.obj,timing=True)
      print "Using FFTW3!"
    except:
      self.useFFTW3=False
      print "FFTW3 not found :-("      
      
  def fft(self,d,inverse=False,timing=False):
    
    if timing: t0 = time.time()
    if self.useFFTW3:
      self.fftw_data[:] = fftshift(d.copy()) # it gives less jumpy phase due to "standard order" of the FFT (A=fft(a) => A[0] is the zero frequency, A[:n/2] contain positive freq, A[n/2+1:] contain the negative freq: http://docs.scipy.org/doc/numpy/reference/routines.fft.html#background-information)
      if inverse: 
          self.plan_ifft()
      else:
          self.plan_fft()            
      out = fftshift(self.fftw_data.copy())          
      
    else:
      if inverse: 
          out = ifftshift(ifft2(ifftshift(d)))
      else: 
          out = fftshift(fft2(fftshift(d)))
          
    if timing:print "FFT time (fftpack):%5.3fs"%(time.time()-t0)
    return out / sqrt(d.size) # normalization of the fft in order to fulfill Parseval thoerem: A=fft(a) -> (A*A.conjugate()).sum() == (a*a.conjugate()).sum() and ifft(A)==a

  def Normalize(self,method="probe"):
      if method == "obj": # normalize L1 norm of the object to 1
          s = abs(self.obj).sum()
          self.obj /= s
          self.probe *= s
      elif method == "probe": # normalize L1 norm of the probe to 1
          s = abs(self.probe).sum()
          self.probe /= s
          self.obj *= s
      elif method == "thr": #threshold on the maxium values of the modulus of hte object
          mask = abs(self.obj)>1
          self.obj[mask] = 1
    
  def CalcForward(self,dx=None,dy=None,frame_index=None):
    """Calculate data from the object and probe,
    If dx==None and dy==None, calculation is done for all
    positions (frame_index) in self.views, and stored there.
    Otherwise calculation is done for a single position.
    """
    if (dx!=None) and (dy!=None):
      psi0 = np.zeros(self.probe.shape).astype(self.DTYPE_CPLX)
      ftpsi0 = np.zeros(self.probe.shape).astype(self.DTYPE_CPLX)
      nx,ny = self.probe.shape
      cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
      psi0 = self.obj[cx:cx+nx,cy:cy+ny]*self.probe # the first index is the vertical direction - top @ 0, second is the horiznontal - left @ 0      
      ftpsi0 = self.fft(psi0)
      return ftpsi0,psi0
    else:        
      if frame_index == None: # Thibault 2009
          frame_range = [0,len(self.views)]
      else: # Maiden2009
          frame_range = [frame_index,frame_index+1]
          
      for d in self.views[frame_range[0]:frame_range[1]]: # This will be done for all frames for Thibault, and for one frame for Maiden
              d.s_calc0,d.psi0 = self.CalcForward(dx=d.dx,dy=d.dy)
              
  def CalcForward_SubPix(self,dx=None,dy=None,frame_index=None):
    """
    Calculate data from the object and probe. Subpixel shift.
    """
    from imProc import shift
    print "sub pix (CalcForward)"    
    if (dx!=None) and (dy!=None):
      psi0 = np.zeros(self.probe.shape).astype(self.DTYPE_CPLX)
      ftpsi0 =  np.zeros(self.probe.shape).astype(self.DTYPE_CPLX)
      nx,ny = self.probe.shape
      cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
      obj_shift = shift(self.obj,-dx,-dy,verbose = False)
      psi0 = obj_shift[cx:cx+nx,cy:cy+ny]*self.probe # the first index is the vertical direction - top @ 0, second is the horiznontal - left @ 0      
      ftpsi0 = self.fft(psi0)
      return ftpsi0,psi0
    else:
      if frame_index == None: # Thibault 2009
          frame_range = [0,len(self.views)]
      else: # Maiden2009
          frame_range = [frame_index,frame_index+1]
      for d in self.views[frame_range[0]:frame_range[1]]:
              d.s_calc0,d.psi0 = self.CalcForward_SubPix(dx=d.dx,dy=d.dy)                 


  def CalcBackward(self,mask=None,frame_index=None, DM = False):
    "The computed modulus of the diffraction pattern is replace by the measured. If mask of valid pixels (0s indicate invalid pixels) is provided, the missing measured values are replaced by the computed ones."
    if frame_index == None:
        frame_range = [0,len(self.views)] # For Thibault2009
    else:
        frame_range = [frame_index,frame_index+1] # For Maiden2009 method

    for d in self.views[frame_range[0]:frame_range[1]]:
        if mask == None:
            tmp = np.abs(d.s_obs)
        else:
            norm = (d.s_obs*mask).sum()/(np.abs(d.s_calc0)*mask).sum() # normalisation of the modulus, othervise it diverges!
            d.s_calc0_norm = d.s_calc0*norm
            imFill = np.abs(d.s_calc0_norm)
            tmp = (mask)*np.abs(d.s_obs)+(1-mask)*imFill # the missing pixels are replaced by the calculated values or by filtered values

        if DM: #Difference Map update of psi for Thibault2009 (testing)
            if not hasattr(d,'s_calc'): d.s_calc = d.s_calc0 # this is for the firs time, otherwise is not defined
            if not hasattr(d,'psi'): d.psi = d.psi0 # this is for the firs time, otherwise is not defined        
            s_tmp = 2*d.s_calc0 - d.s_calc            
            d.s_calc = tmp*exp(1j*np.angle(s_tmp))
            d.psi = d.psi + self.fft(d.s_calc,inverse=True) - d.psi0 # Difference map update of psi
        else:            
            d.s_calc = tmp*exp(1j*np.angle(d.s_calc0))
            d.psi = self.fft(d.s_calc,inverse=True)

  def UpdateObject(self,frame_index=None):
    """
    This is equation from setting gradient of the square-of differences between the measured 
    and estimatedwrt to Object to zero.  
    """
    nx,ny = self.probe.shape        
    if "thibault2009" in self.method.lower():
      obj = np.zeros_like(self.obj)
      objnorm = np.zeros_like(abs(self.obj))
      for d in self.views:          
        dx,dy = d.dx,d.dy #o positions of the scan
        cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
        objnorm[cx:cx+nx,cy:cy+ny] += abs(self.probe)**2
        obj[cx:cx+nx,cy:cy+ny] += self.probe.conjugate()*d.psi
      if "max" in self.method.lower(): #This is for Thibault2009-max
          self.obj = obj/objnorm.max()
      else:       
        if 'reg_const_object' not in self.params.keys():
            self.params['reg_const_object'] = 1e-2        
        reg = self.params['reg_const_object']*objnorm.max() #empirical threshold
        self.obj = obj/(np.maximum(objnorm,reg)) # to avoid underflow
        
    elif "maiden2009" in self.method.lower():        
        if 'learning_const_object' not in self.params.keys():
            self.params['learning_const_object'] = 1 # learing constant, value of a=1 used in maiden2009 paper
        d = self.views[frame_index]
        dx,dy = d.dx,d.dy
        cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
        self.obj[cx:cx+nx,cy:cy+ny] += ePIEupdate(self.probe,d.psi,d.psi0,self.params['learning_const_object'])
    else:
        raise Exception('Unknown method: '+self.method)

  def UpdateObject_SubPix(self,frame_index=None):
    """
    This is equation from setting gradient of the square-of differences between the measured 
    and estimated wrt to Object to zero.  
    """
    from imProc import shift
    print "sub pix (update object)"
    nx,ny = self.probe.shape        
    if "thibault2009" in self.method.lower():
      obj = np.zeros(self.obj.shape).astype(self.DTYPE_CPLX)
      objnorm = np.zeros(self.obj.shape).astype(self.DTYPE_REAL)
      for d in self.views:          
        dx,dy = d.dx,d.dy #o positions of the scan
        obj_tmp = np.zeros_like(self.obj)
        cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
        obj_tmp[cx:cx+nx,cy:cy+ny] = self.probe.conjugate()*d.psi
        self.obj += shift(obj_tmp,dx,dy)
        
        
        objnorm[cx:cx+nx,cy:cy+ny] += abs(self.probe)**2
        obj[cx:cx+nx,cy:cy+ny] += self.probe.conjugate()*d.psi
      if "max" in self.method.lower(): #Thisis for Thibault2009-max
          self.obj = obj/objnorm.max()
      else:
#        reg = 0.01*objnorm.max() #empirical threshold
        
        reg = 0.1*objnorm.max() #empirical threshold
        self.obj = obj/(np.maximum(objnorm,reg)) # to avoid underflow

    elif "maiden2009" in self.method.lower():
        if 'learning_const_object' not in self.params.keys():
            self.params['learning_const_object'] = 1 # learing constant, value of a=1 used in maiden2009 paper
        d = self.views[frame_index]
        dx,dy = d.dx,d.dy
        cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
        self.obj[cx:cx+nx,cy:cy+ny] += ePIEupdate(self.probe,d.psi,d.psi0,self.params['learning_const_object'])
    else:
        raise Exception('Unknown method: '+self.method)

    
  def UpdateProbe(self,frame_index=None):
    nx,ny = self.probe.shape
    if "thibault2009" in self.method.lower():
      probe = np.zeros_like(self.probe)
      probenorm = np.zeros_like(abs(self.probe))
      for d in self.views:
        dx,dy = d.dx,d.dy
        cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
        probenorm += abs(self.obj[cx:cx+nx,cy:cy+ny])**2
        probe += self.obj[cx:cx+nx,cy:cy+ny].conjugate()*d.psi
      if "max" in self.method.lower(): #Thisis for Thibault2009-max
          self.probe = probe/probenorm.max()          
      else: 
          if 'reg_const_probe' not in self.params.keys():
            self.params['reg_const_probe'] = 1e-2        

          reg = self.params['reg_const_probe']*probenorm.max() #emprirical threshold
          self.probe = probe/np.maximum(probenorm,reg) # to avoid underflow

    elif "maiden2009" in self.method.lower():
        if 'learning_const_probe' not in self.params.keys():
            self.params['learning_const_probe'] = 1 # learing constant, value of a=1 used in maiden2009 paper

        d = self.views[frame_index]
        dx,dy = d.dx,d.dy
        cx,cy = get_view_coord(self.obj.shape,self.probe.shape,(dx,dy))
        self.probe += ePIEupdate(self.obj[cx:cx+nx,cy:cy+ny],d.psi,d.psi0,self.params['learning_const_probe'])
    else:
        raise Exception('Unknown method: '+self.method)
          
  def SaveStateOfPsiHIO(self,state=0):      
      for d in self.views:          
          initVal=np.zeros(np.append(1,d.psi0.shape),dtype=self.DTYPE_CPLX)
          if hasattr(d,'psiState')==False: 
              d.psiState=initVal
          while d.psiState.shape[0]<(state+1): 
              d.psiState=np.append(d.psiState,initVal,axis=0)
          d.psiState[state]=d.psi
          if state==1:
              d.psiState[state]=d.psi0              

  def IntermediateUpdate(self):
      for d in self.views:
          #d.psi0=2*d.psiState[1]-d.psiState[0]
          d.psi0 = 2*d.psi0-d.psi

  def FinalUpdate(self):
      for d in self.views:
          d.psi = d.psiState[0]+d.psi-d.psiState[1]
  
  def R(self, i=None, chi2=False, mask=None, return_views=False):
    if mask == None: 
        mask = 1.                
    if i!=None:
      calc = (abs(self.views[i].s_calc0)**2)*mask
      obs = (abs(self.views[i].s_obs)**2)*mask
      #scalef = (calc*obs).sum()/(calc**2).sum()
      scalef = 1
      tmpR = (abs(calc*scalef-obs)**2).sum()
      rnorm = (obs**2).sum()
    else:
      tmpR,rnorm = 0,0
      n_views = len(self.views)
      R_view = np.zeros(n_views)
      rnorm_view = np.zeros(n_views)      
      for i in xrange(n_views):
        R_view[i], rnorm_view[i]= self.R(i,chi2=True,mask=mask)
              
      tmpR += R_view.sum()
      rnorm += rnorm_view.sum()      
    if chi2: 
        if return_views:
            return tmpR, rnorm, R_view, rnorm_view        
        else:
            return tmpR,rnorm            
    else: 
        if return_views:
            return tmpR/rnorm, R_view/rnorm_view
        else:            
            return tmpR/rnorm
    
        
  def PlotData(self,i,dlogs=4):
    plt.subplot(121)
    plt.imshow(log10(abs(self.views[i].s_obs)),vmin=log10(abs(self.views[i].s_obs)).max()-dlogs)
    plt.title("$Obs[%d]$"%i)
    plt.subplot(122)
    plt.imshow(log10(abs(self.views[i].s_calc0)),vmin=log10(abs(self.views[i].s_calc0)).max()-dlogs)
    plt.title("$Calc[%d]$"%i)    
    
  def PlotObject(self):
    plt.imshow(complex2rgbalin(self.obj))

  def PrintProgress(self,r, i, dt):
      print "Ptycho cycle %3d: R = %6.4e%%,  dt/cycle = %6.2fs"%(i,r*100,dt)
      s_calc_logabs = np.log10(abs(self.views[0].s_calc0).clip(1e-6))*exp(1j*np.angle(self.views[0].s_calc0))
      s_obs0 = np.log10(self.views[0].s_obs.clip(1e-6))
      showProgress(s_calc_logabs,s_obs0,tit1='s_calc0',tit2='s_obs',figNum=101, stit="%s: cycle %3d: R= %6.4f%%"%(self.method,i,r*100))  
      showProgress(self.obj,self.probe, stit="%s: cycle %3d: R= %6.4f%%"%(self.method,i,r*100))      
                            
  def Run(self, ncycle, method="Thibault2009", updateProbe=False, verbose=False, mask=None, subPix = False, params=None):
    """
    ncycle: number of iterations
    method:
    Thibalut2009: Thibault P, Dierolf M, Bunk O, Menzel A, Pfeiffer F (2009) Probe retrieval in ptychographic coherent diffractive imaging. Ultramicroscopy 109:338-343.      
    Maiden2009: Maiden, A. M., & Rodenburg, J. M. (2009). An improved ptychographical phase retrieval algorithm for diffractive imaging. Ultramicroscopy, 109(10), 1256-62.
    updateProbe: set to True to update probe in each iteration
    verbose: set to True to print updates on the convergence, set to N>1 to plot the progres every N-steps
    mask: insert the mask of valid pixels (bad, invalid or zero pixels are zeros)
    params: dicitionary with certain parameters for evaluation
    """
    if params is None:
        params = {}
    if isinstance(method, basestring):
        self.method = method
        self.params = {'method':method}
    elif isinstance(method, dict):
        self.method = method['method']
        self.params = method
    else:
        raise Exception('Unknown format of the method parameters')
        
    print "Using method:", self.method      
    print "Evaluating %d frames."%len(self.views)
    self.mask = mask
    if self.mask!=None: print "Using mask!"
    else: print "No mask!"
    if not hasattr(self, 'Rarray'): 
        self.Rarray=[]
    
    if 'thibault' in self.method.lower():        
        for i in xrange(ncycle):
          t0=time.time()
          #o multiplies object * probe
          if subPix: 
              self.CalcForward_SubPix() 
          else:               
              self.CalcForward() 
              
          #o replaces the modulus of the estimated diffraction by the measured
          self.CalcBackward(mask=self.mask,DM = ('DM' in self.method)) 
          # updates of the object and the probe
          innerUpdates=1      
          for j in range(0,innerUpdates):    
              if subPix: 
                  self.UpdateObject_SubPix(method=method)
              else: 
                  self.UpdateObject()
              if updateProbe: 
                  self.UpdateProbe()
          #self.Normalize(method="probe") # keep probe nomalied: probe.sum()=1
          self.Rarray=np.append(self.Rarray,self.R(mask=self.mask))
          if verbose:
            if i%verbose==0:
                self.PrintProgress(self.Rarray[-1],i,time.time()-t0)
                
    elif 'maiden' in self.method.lower():
        for i in xrange(ncycle):
            t0 = time.time()
            randSeq=np.random.permutation(np.arange(0,len(self.views)))
            count=0
            for j in randSeq:
                count += 1
                self.CalcForward(frame_index=j)
                self.CalcBackward(frame_index=j,mask=self.mask)
                self.UpdateObject(frame_index=j)
            if updateProbe:
                randSeq = np.random.permutation(np.arange(0,len(self.views)))
                for j in randSeq:
                    self.CalcForward(frame_index=j)
                    self.CalcBackward(frame_index=j,mask=self.mask)
                    self.UpdateProbe(frame_index=j)
            self.Rarray = np.append(self.Rarray,self.R(mask=self.mask))
            if i%verbose==0:
                self.PrintProgress(self.Rarray[-1],i,time.time()-t0)


  def RunHIO(self,ncycle,method="Thibault2009",updateProbe=False,verbose=False):
    print "Using method:", method  
    self.CalcForward() #o multiplies object * probe
    self.CalcBackward() #o replaces the modulus of the estimated diffraction by the measured
    for i in xrange(ncycle):
      t0=time.time()
      #self.CalcForward() #o This only splits the current object into views and multiplies object * probe      
      self.SaveStateOfPsi(state=0) # save psi      
      innerUpdates=1      
      for j in range(0,innerUpdates):    
          self.UpdateObject(method=method)
          if updateProbe: 
              if ((i+1)%updateProbe)==0:
                  self.UpdateProbe(method=method)

      self.CalcForward()
      self.SaveStateOfPsi(state=1) # save psi0
      self.IntermediateUpdate()
      self.CalcBackward()
      self.FinalUpdate()

      if verbose:
        print "Ptycho cycle %3d: R= %6.4f%%,  dt=%6.2fs"%(i,self.R()*100,time.time()-t0)
        if True:
              showProgress(self.obj,self.probe)

  def RefinePos(self, method='Powell', opt={'disp':True, 'maxfeval':10}):
      """
      Refines positions of the ptychograhic dataset.
      """
      print "Refinement of the scan positions (using %s)."%(method)
      print "You can stop at any time with 'ctrl+C', the results are stored in the object p.vews[i].dx, p.views[i].dy"
      print "Should be followed by p.Run(...) to update probe and obj to the new positions."

      px,py = get_pos(self)
      args = self,
      res = minimize(R_pos, np.concatenate( (px,py) ,axis=0) , args=args, method=method, options=opt)
      return res.x.reshape(2,-1)

  def RefinePosUpdate(self, n_cycles=3, method_refine='Powell', opt={'disp':True, 'maxfeval':10}, method_ptycho='Maiden2009', n_updates=100):
      """
      Refines positions of the ptychograhic dataset and cycles it with ptychographic updates. 
      """
      print "Refinement of the scan positions (using %s minimisation). %d cycles alternated with %s update to adjust both obj and probe. "%(method_refine, n_cycles,method_ptycho)

      for i in range(n_cycles):
          self.RefinePos(self, method = method_refine, opt=opt)
          self.Run(n_updates, verbose=10, updateProbe=True, method=method_ptycho)

  def DirectMin(self, which='obj', noise='gauss', reg_fac = 0., verbose=True, maxiter=1000):
      """
      Conjugate gradient based direct minimisation.
      """
      print "Direct minimisation of %s, with regularisation factor %1.2e."%(which,reg_fac)
      print "You can stop at any time with 'ctrl+C', the results are stored in the object p (p.obj, p.probe)."
      if which is 'obj':
          resCG = minimize(directmin_f, image_flatten(self.obj), args=(self, which, noise, reg_fac), method='CG', jac=directmin_f_der,options={'disp': True, 'maxiter': maxiter})
      else:
          print "todo: To be properly tested... "
          pass
      return resCG
      
    
  def SaveResults(self, resdir="./Results/", name_appendix = ""):
        if not(os.path.isdir(resdir)): os.mkdir(resdir)  
        print "Saving results to %s/<variable_name>%s.npy"%(resdir,name_appendix)
        np.save(resdir+'/obj' + name_appendix,self.obj)
        np.save(resdir+'/probe' + name_appendix,self.probe)
        np.save(resdir+'/obj0'+name_appendix,self.obj0)
        np.save(resdir+'/probe0'+name_appendix,self.probe0)
        np.save(resdir+'/R'+name_appendix,self.Rarray)    

################################################
################################################
#
# Refinement of the scan positions
#
################################################
################################################

def R_pos(x, p):    
    xr = x.reshape(2,-1);
    dlim = 10 # this is to avoid too large steps which can lead to loss of the scan position...
    for i,d in enumerate(p.views):
        if (abs(d.dx-xr[0,i]) < dlim) and (abs(d.dy-xr[1,i]) < dlim):
            d.dx,d.dy = xr[0,i],xr[1,i]
    
    #p.Run(1,verbose=0,updateProbe=True,method='Thibault2009')
    p.CalcForward()
    out = p.R(chi2=True)[0]
    print 'R = %.4e'%out
    return out
    

################################################
################################################
#
# Direct minimisation
#
################################################
################################################

def image_flatten(im):
    """
    Flattens complex object into one dimensional vectro - first half is the real part, second half is the imaginary part. Inverse operation by image_reshape. 
    """
    im_real = im.real.flatten()
    im_imag = im.imag.flatten()
    return np.concatenate( (im_real,im_imag) )

def image_reshape(x,x_shape):
    """
    Reshapes one-dimensional vector obtained by image_flatten back to a 2D complex image.
    """
    xr = x.reshape(2,-1);
    im_real = xr[0].reshape(x_shape)
    im_imag = xr[1].reshape(x_shape)
    return im_real + 1j*im_imag

#todo: make p.R() work for Poisson noise as well
def directmin_f(x,p,which='obj',noise='gauss', reg_fac=0, verbose=True):        
    if which == 'both':
        no = 2*np.prod(p.obj.shape)
        p.obj = image_reshape(x[:no],p.obj.shape)
        p.probe = image_reshape(x[no:],p.probe.shape)            
    elif which == 'obj':
        p.obj = image_reshape(x,p.obj.shape)            
    elif which == 'probe':
        p.probe = image_reshape(x,p.probe.shape)
    p.CalcForward()
    out = p.R(chi2=True)[0]
    if reg_fac>0:
        out += reg_fac*abs(reg(x,p.obj.shape)).astype(out.dtype)        
    if verbose: 
        print 'R = %.4e'%out
    return out
    
def directmin_f_der(x,p,which='obj',noise='gauss',reg_fac=0):
    if which == 'both':
        no = 2*np.prod(p.obj.shape)
        p.obj = image_reshape(x[:no],p.obj.shape)
        p.probe = image_reshape(x[no:],p.probe.shape)            
    elif which == 'obj':
        p.obj = image_reshape(x,p.obj.shape)            
    elif which == 'probe':
        p.probe = image_reshape(x,p.probe.shape)
    ###
    der_obj = np.zeros_like(p.obj)
    der_probe = np.zeros_like(p.probe)
    px, py = p.probe.shape
    posx,posy = get_pos(p)
    for v in p.views:        
        intensity_cal = abs(v.s_calc0)**2 
        intensity_obs = abs(v.s_obs)**2 
        if noise.lower() == 'gauss':
            weird_factor = 2 # this should be one according to math but does not fit with approximated gradient. 
            chi = weird_factor*2*v.s_calc0*(intensity_cal - intensity_obs)
        elif noise.lower() == 'poisson':
            weird_factor = 2 # this should be one according to math but does not fit with approximated gradient. same as for gauss!!!
            chi = weird_factor*v.s_calc0*(intensity_obs - intensity_cal)/intensity_cal
        
        cx,cy = get_view_coord(p.obj.shape,p.probe.shape,(v.dx,v.dy))
        ftchi = p.fft(chi,inverse=True)
        if which in ('obj', 'both'):
            der_obj[cx:cx+px,cy:cy+py] += p.probe * ftchi.conjugate()
        if which in ('probe','both'):
            der_probe += p.obj[cx:cx+px,cy:cy+py] * ftchi.conjugate()

    if which in ('obj', 'both'):    
        der_obj = der_obj.conjugate() # not sure why there is conjugate but with this it agrees with the numerical approximation
    if which in ('probe', 'both'):            
        der_probe = der_probe.conjugate()
    
    if which == 'both':
        ders = np.concatenate( (image_flatten(der_obj), image_flatten(der_probe)) ) 
    elif which == 'obj':
        ders = image_flatten(der_obj)        
    elif which == 'probe':
        ders = image_flatten(der_probe)  
    if reg_fac>0:
        ders += reg_fac*reg_der(x,p.obj.shape)
        
    return ders

def reg(x,sh):
    """
    Regularisation penalty for smoothness of the images in direct minimisation.
    """
    A = image_reshape(x,sh)
    d1 = A[:-1,:]-A[1:,:]
    d2 = A[:,:-1]-A[:,1:]
    s1 = (d1*d1.conjugate()).sum()
    s2 = (d2*d2.conjugate()).sum()
    return s1+s2
    
def reg_der(x,sh):
    A = image_reshape(x,sh)
    Ac = A # according to my math there should be conjugate here (but this agrees with numerical approx), similar to directmin_f_der
    A1 = np.zeros_like(Ac)
    A2 = np.zeros_like(Ac)
    A3 = np.zeros_like(Ac)
    A4 = np.zeros_like(Ac)
    A1[:-1,:] = Ac[1:,:]
    A2[1:,:] = Ac[:-1,:]
    A3[:,:-1] = Ac[:,1:]
    A4[:,1:] = Ac[:,:-1]
    weird_factor = 2 # according to my math this should be 1 (but this agrees with numereical approx) - this is the same as in the directmin_f_der
    return weird_factor*image_flatten(4*Ac-A1-A2-A3-A4)
    
def directmin_optimal_reg_factor(p):
    fd = directmin_f_der(image_flatten(p.obj), p)
    rd = reg_der(image_flatten(p.obj), p.obj.shape)
    rat = abs(fd).max()/abs(rd).max()
    nphot = 0
    for d in p.views:
        nphot += d.s_obs.sum() #total number of photons
    rec = 0.01*p.obj.size**2/(p.views[0].s_obs.size*len(p.views)*nphot) # estimate from \cite{Thibault2012}
    print 'Recomanded reg_fac = %.2e\n'%rec,    
    print 'Equality of derivatives for reg_fac = %.2e'%rat

    return rec, rat
################################################
################################################
#
# Simulation of the ptychographic data.
# Requires imProc module:
# git clone git@github.com:aludnam/imProc.git
#
# For visualisation requires module plottools:
# git clone git@github.com:aludnam/plottools.git
#
################################################
################################################
        
import Image, imProc, plotting        
def get_img(index=0):    
    """
    Returns image (numpy array) from the path_list.
    
    To do: use import skimage to get images!
    """    
    path = os.path.dirname(sys.modules[__name__].__file__)
    path_list = "/Images/lena.tif", "/Images/devoogd.tif", "/Images/erika.tif"
    im = Image.open(path+path_list[index]) 
    return np.array(im)
    
def gauss2D(im_size=(64,64),mu=(0,0),sigma=(1,1)):
    """
    2D gaussian function normalised to 1.
    The first coordinate is vetical, second is horizontal!    
    """
    nx, ny = im_size
    x = np.linspace(-np.fix(nx/2),np.fix(nx/2),nx)
    y = np.linspace(-np.fix(ny/2),np.fix(ny/2),ny)
    yy,xx = np.meshgrid(y,x) # the first coordinate is vetical, second is horizontal!
    v = np.array(sigma)**2
    g = 1/(2*np.pi*np.sqrt(v[0]*v[1])) * np.exp(-((xx-mu[0])**2/(2*v[0]) + (yy-mu[1])**2/(2*v[1])))        
    return g 

def make_beam_stop(im_size=(256,256), radius = 10):
    nx, ny = im_size
    x = np.linspace(-np.fix(nx/2),np.fix(nx/2),nx)
    y = np.linspace(-np.fix(ny/2),np.fix(ny/2),ny)
    yy,xx = np.meshgrid(y,x) # the first coordinate is vetical, second is horizontal!
    return xx**2 + yy**2 >= radius**2


def spiral_archimedes(a,n):
  """" Creates np points spiral of step a, with a between successive points
  on the spiral. Returns the x,y coordinates of the spiral points.
  
  This is an Archimedes spiral. the equation is:
    r=(a/2*pi)*theta
    the stepsize (radial distance between successive passes) is a
    the curved absciss is: s(theta)=(a/2*pi)*integral[t=0->theta](sqrt(1*t**2))dt
  """
  vr,vt=[0],[0]
  t=np.pi
  while len(vr) < n:
    vt.append(t)
    vr.append(a*t/(2*np.pi))
    t+=2*np.pi/np.sqrt(1+t**2)
  vt, vr = np.array(vt), np.array(vr)
  return vr*np.cos(vt), vr*np.sin(vt)


def spiral_fermat(dmax,n):
  """" 
  Creates a Fermat spiral with n points distributed in a circular area with 
  diamter<= dmax. Returns the x,y coordinates of the spiral points. The average
  distance between points can be roughly estimated as 0.5*dmax/(sqrt(n/pi))
  
  http://en.wikipedia.org/wiki/Fermat%27s_spiral
  """
  c=0.5*dmax/np.sqrt(n)
  vr,vt=[],[]
  t=.4
  goldenAngle = np.pi*(3-np.sqrt(5))
  while t<n:
    vr.append(c*np.sqrt(t))
    vt.append(t*goldenAngle)
    t += 1
  vt, vr = np.array(vt), np.array(vr)
  return vr*np.cos(vt),vr*np.sin(vt)

def psi(obj, probe, shift_val, border):  
    shift_int = np.round(shift_val)
    shift_sub = shift_val - shift_int
    ocx, ocy = obj.shape[0]//2, obj.shape[1]//2
    pcx, pcy = probe.shape[0]//2, probe.shape[1]//2
    sxr, syr = shift_int[0]+ocx-pcx, shift_int[1]+ocy-pcy
    
    ox1, ox2 = sxr - border, sxr + probe.shape[0] + border 
    oy1, oy2 = syr - border, syr + probe.shape[1] + border 
    
    return probe*imProc.shift(obj[ox1:ox2,oy1:oy2], shift_sub[0], shift_sub[1],verbose=False)[border:-border,border:-border]
    
class Im: 
    def __init__(self,values=None, info={}):
        self.values = values
        self.info = info
        if values != None:
            self.info.update({'type': 'Custom'})
    
    def make_even(self):
        "Ensures even shape of the images:"
        if len(self.values) > 2 and np.ndim(self.values) < 3: # does not work for scan positions or for stack of images
            psx,psy = self.values.shape
            if psx%2: psx -= 1
            if psy%2: psy -= 1
            if (psx,psy) != self.values.shape:
                self.values = self.values[:psx,:psy]
                self.info['shape'] = self.values.shape
                self.info['type'] += 'Even' 
                print "Changing shape to even values: (%g,%g)"%self.values.shape                        
                
    def show(self):
        if len(self.values) == 2: # scan poisitions
            plt.figure()
            plt.plot(self.values[1],self.values[0],'-x')
            plt.axis('equal')
            plt.gca().invert_yaxis()
            plt.xlabel('pos_hor [pixels]')
            plt.ylabel('pos_ver [pixels]')
            plt.grid(b=True)
        elif np.ndim(self.values) < 3: # for 2D complex images
            plotting.showCplx(self.values)
        else:
            pass
            
class Simulation:
    """
    Simulation of the ptychographic data.
    """
    def __init__(self, obj=None, obj_info={}, probe=None, probe_info={}, scan=None, scan_info={}, data_info={}, verbose=1, prec='single'):
        """
        obj_info: dictionary with   obj_info['type'] = ('real','real_imag','ampl_phase') # type of the object
                                    obj_info['phase_stretch'] = ps #  specify the stretch of the phase image (default is ps=2pi)
                                    obj_info['alpha_win'] = a # ratio of the obj edge to be dimmed down by imProc.tukeywin2D. This is to artefact from sharp edges of the object. 
                                    defaults set in update_default_obj
                                    
        obj: specific value of obj can be passed (2D complex numpy array). obj_info can be passed as empty dictionary (default).
            
        probe_info: dictionary with probe_info['type'] = ('flat', 'Gauss',) # type of the probe
                                    probe_info['size'] = (sizex,sizey) # size of the porbe
                                    probe_info['sigma_pix'] = value # sigma for gaussian probe
                                    defaults set in update_default_probe
                                    
        probe: specific value of probe can be passed (2D complex numpy array). probe_info can be passed as empty dicitionary (default).

        scan_info: dictionary with  scan_info['type'] = ('spiral' ,'raster') # type of the scan 
                                    scan_info['scan_step_pix'] = value # step in the scan in [nm]
                                    scan_info['n_scans'] = value # number of scan positions
                                    scan_info['integer_values'] = True # integer values of the scan positions
                                    defaults set in update_default_scan
                                    
        scan: specific value of scan can be passed (list/tuple of two arrays (posx, posy)). scan_info can be passed as empty dicitionary (default).
                                    
        data_info: dictionary with data_info['pix_size_direct_nm'] = value # pixels size in the object space [nm]
                                    data_info['num_phot_max'] = value #number of photosn in maximum pixel
                                    data_info['bg'] = value # uniform background added to each frame
                                    data_info['beam_stop_tranparency'] = value
                                    data_info['noise'] = 'poissoin' will create Poisson noise in the data. None will make noise-free data.
                                    defaults set in update_default_data
                                    
        EXAMPLES:
        d = ptycho.Simulation() # Initialises simulation object with default values of probe, object and scan
        
        or        

        d = ptycho.Simulation(obj=obj, probe=probe, scan=scan) # Initialises ptychographic dataset with specifice obj (complex numpy array), probe (complex numpy array) and scan (list of two numpy arrays specifing verical and horizontal position of the can in pixels)
        
        or 
        
        d = ptycho.Simulation(obj_info={'num_phot_max':10**3}, probe_info={'type':'Gauss','sigma_pix':(50,30)}, scan_info={'type':'spiral','n_scans':77}) # Initialises simulation object with specific values of obj and probe and scan, missing parameters are filled with default values
        
                
        d.make_data() # creates obj, probe, scan and ptychographic dataset
        d.make_obj() # creates obj only
        d.make_probe() # crete probe only
        d.print_info() # prints all the parameters of the simulation
                
        """
        if prec is 'single':    # precision of the computation
            self.DTYPE_REAL = np.float32    #dtype of the real arrays
            self.DTYPE_CPLX = np.complex64    #dtype of the complex arrays
        elif prec is 'double':
            self.DTYPE_REAL = np.float64
            self.DTYPE_CPLX = np.complex128
        print 'Using %s precision.'%prec
        
        self.obj = Im(obj, obj_info)
        self.probe = Im(probe, probe_info)
        self.scan = Im(scan,scan_info)
        self.data_info = data_info
        self.verbose = verbose
                        
    def update_default_data(self):
        print 'Updating defaults values for simulation.'
        default_data_info = {'pix_size_direct_nm': 10,
                       'num_phot_max': 1e9, 
                       'bg': 0, 
                       'beam_stop_transparency': 0,
                       'noise': 'poisson'}
                       
        default_data_info.update(self.data_info)
        self.data_info = default_data_info.copy()

    def update_default_scan(self):
        print 'Updating defaults values for scan.'
        default_scan_info = {'type': 'spiral',
                       'scan_step_pix': 30,
                       'n_scans': 50, 
                       'integer_values': True}
        default_scan_info.update(self.scan.info)
        self.scan.info = default_scan_info.copy()
        
    def update_default_obj(self):
        print 'Updating defaults values for object.'
        default_obj_info = {'type': 'ampl_phase'}
        default_obj_info.update(self.obj.info)                         
        self.obj.info = default_obj_info.copy()

    def update_default_probe(self):
        print 'Updating defaults values for probe.'        
        default_probe_info = {'type': 'gauss', 
                           'shape': (256,256), 
                           'sigma_pix': (50,50)} # in pixels
        default_probe_info.update(self.probe.info)                                                    
        self.probe.info = default_probe_info.copy()
    
    def print_info(self):
        print "Parameters of the simulation:"
        print "Data info:", self.data_info
        print "Scan info:", self.scan.info
        print "Object info:", self.obj.info
        print "Probe info:", self.probe.info


    def make_data(self):
        
        self.update_default_data()
        
        if self.obj.values == None:
            self.make_obj()                            
        
        if self.probe.values == None: 
            self.make_probe()

        if self.scan.values == None: 
            self.make_scan()        
        
        if self.verbose: 
            print "Simulating ptychographic data."                 
        self.obj.make_even()
        self.probe.make_even()
        
        posx, posy = self.scan.values
        posx_max, posy_max = np.ceil(abs(posx).max()), np.ceil(abs(posy).max())    
        n = len(posx) # number of scans
        s_v = self.probe.values.shape
        if 'rebin_factor' in self.data_info:
            rf = self.data_info['rebin_factor']            
        else:
            rf = 1
        s_a = s_v[0]/rf, s_v[1]/rf                    
        self.make_obj_true(posx_max,posy_max) # pads the obj with zeros if necessary        
        self.psi = Im(np.zeros((n, s_v[0], s_v[1]), dtype = self.DTYPE_CPLX))        
        intensity = np.zeros((n, s_a[0], s_a[1]), dtype = self.DTYPE_REAL)
        if 'beam_stop_radius' in self.data_info:
            if 'beam_stop_transparency' not in self.data_info:
                self.data_info['beam_stop_transparency'] = 0
            if self.verbose: print "Beam stop with transparency", self.data_info['beam_stop_transparency']
            beam_stop_tmp = make_beam_stop(self.probe.values.shape, self.data_info['beam_stop_radius'])
            beam_stop = self.data_info['beam_stop_transparency']*(1-beam_stop_tmp) + beam_stop_tmp
        else: 
            beam_stop = 1

        for sx, sy, index in zip(posx, posy, range(n)):
            sys.stdout.write('.')
            self.psi.values[index] = psi(self.obj_true.values, self.probe.values, (sx, sy), self.obj_true.info['border_pix'])
            intensity_tmp = abs(np.fft.fftshift(np.fft.fft2(self.psi.values[index])))**2
            intensity_tmp *= beam_stop                            
            
            if rf > 1: # rebin_factor
                intensity[index] = imProc.rebin(intensity_tmp,rf)            
                if (self.verbose and index == (len(posx)-1)): print "\nBinning data by", rf
            else:
                intensity[index] = intensity_tmp
                        
        print("\n")             
        
        intensity /= intensity.max()
        intensity *= self.data_info['num_phot_max']
        intensity += self.data_info['bg']
        if self.data_info['noise'] == 'poisson':
            intensity= np.random.poisson(intensity)
        self.amplitude = Im(np.sqrt(intensity))

        if self.verbose:
            self.print_info()


    def make_obj_true(self, posx_max, posy_max):
        info = {'border_pix':2} # enlarge on each side
        o_s = self.obj.values.shape
        p_s = self.probe.values.shape
        ot_s = 2*(posx_max + (p_s[0]+1)//2 + info['border_pix']), 2*(posy_max + (p_s[1]+1)//2 + info['border_pix'])
        ot_s = max(ot_s[0],o_s[0]), max(ot_s[1],o_s[1]) # the case of very large object ot_s<o_s
        tmp = np.zeros(ot_s, np.dtype(self.obj.values.dtype))
        start = ot_s[0]//2 - o_s[0]//2, ot_s[1]//2 - o_s[1]//2
        tmp[start[0]:start[0] + o_s[0],start[1] : start[1] + o_s[1]] = self.obj.values
        self.obj_true = Im(tmp,info)

    def make_obj(self):
        self.update_default_obj()
        info = self.obj.info
        obj_type = info['type'].lower()
        print "Simulating object:",obj_type

        if obj_type == 'custom':
            obj = self.obj.values
            
        elif ('ampl' in obj_type) and ('phase' in obj_type):
            im0 = self.DTYPE_REAL(get_img(0))
            im1 = self.DTYPE_REAL(get_img(1))
            # Strethc the phase to interval (-phase_stretch/2, +phase_stretch/2)
            phase0 = im1 - im1.min()
            ps = 2*np.pi
            phase_stretch = ps*phase0/self.DTYPE_REAL(phase0.max()) - ps/2.        
            obj = im0 * np.exp(1j*phase_stretch)
            
        elif ('real' in obj_type) and ('imag' in obj_type):
            im0 = self.DTYPE_REAL(get_img(0))
            im1 = self.DTYPE_REAL(get_img(1))
            obj = im0 + 1j*im1
            
        elif obj_type.lower() == 'random':
            s = info['shape']
            rand_phase = 2*np.pi*np.random.rand(s[0],s[1])-np.pi
            obj = np.random.rand(s[0],s[1])*np.exp(1j*rand_phase)
            
        elif obj_type.lower() == 'flat':
            obj = np.ones(info['shape']).astype(self.DTYPE_CPLX)
            
        else:
            msg = "Unknown object type:", self.obj.info['type']
            raise NameError(msg)            

        if 'phase_stretch' in info:         
            phase = np.angle(obj)
            phase0 = phase - phase.min()                        
            if phase0.any():                
                ps = info['phase_stretch']            
                phase_stretch = ps * phase0/self.DTYPE_REAL(phase0.max()) - ps/2.
                obj = abs(obj) * np.exp(1j*phase_stretch)
        
        if 'alpha_win' in info:
            s =  obj.shape
            w = imProc.tukeywin2D(s, info['alpha_win'])    
            obj *= w # tuckey window to smooth edges (alpha_win)
            
        self.obj.values = self.DTYPE_CPLX(obj)

    def make_probe(self):
        """
        Simulates the beam.
        """      
        self.update_default_probe()
        info = self.probe.info
        print "Simulating the beam:",info['type']
        probe_type = info['type'].lower()
        
        if probe_type == 'custom':
            pass
        elif probe_type == 'flat':
            self.probe.values = self.DTYPE_CPLX(np.ones(info['shape']))
            
        elif probe_type == 'gauss':
            self.probe.values = self.DTYPE_CPLX(gauss2D(info['shape'], mu = (0,0), sigma = info['sigma_pix']))

        else: 
            msg = "Unknown probe type:", self.probe.info['type']
            raise NameError(msg)
            

    def make_scan(self):
        self.update_default_scan()        
        info = self.scan.info
        print "Simulating scan:",info['type']
        scan_type = info['type'].lower()        
        if scan_type == 'rect':
            scanX = np.linspace(0,self.obj.values.shape[0],info['scan_step_pix'])
            scanY = np.linspace(0,self.obj.values.shape[1],info['scan_step_pix'])        
            if scanX.size==0: scanX = np.array([0]) # to make at least one scan
            if scanY.size==0: scanY = np.array([0])            
            posx = scanX - scanX.mean()
            posy = scanY - scanY.mean()
            
        elif scan_type == 'spiral':        
            posx,posy = spiral_archimedes(info['scan_step_pix'],info['n_scans'])                
        elif scan_type =='spiralfermat':
            posx,posy = spiral_archimedes(info['scan_step_pix'],info['n_scans'])                    
        else: 
            msg = "Unknown scan type:", self.scan.info['type']
            raise NameError(msg)
            
        if info['integer_values']:
            print "Integer values of the scan!"
            self.scan.values = (posx.round(),posy.round())
        else:
            self.scan.values = (posx,posy)
            
    def show_illumination_sum(self, log_values = False):
        posx, posy = self.scan.values
        illum = np.zeros_like(self.obj_true.values)        
        nix,niy = illum.shape
        npx,npy = self.probe.values.shape
        for sx, sy in zip(posx, posy):
            startx = nix/2+sx-npx/2
            starty = niy/2+sy-npy/2
            illum[startx:startx+npx,starty:starty+npy] += self.probe.values
        #plotting.showCplx(illum)
        plt.figure()
        if log_values:
            im2show = np.log10(abs(illum))
        else:
            im2show = abs(illum)
        plt.imshow(im2show,interpolation='Nearest',extent=(0,illum.shape[1],0,illum.shape[0]))
        plt.plot(posy+niy/2,-posx+nix/2,'*:k') # for "plot" the vetical axis is inverted compared to "imshow"
        plt.plot(posy+niy/2,-posx+nix/2,'xw')
        return illum