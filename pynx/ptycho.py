########################################################################
#
# Ptychographic Reconstruction
#
#  Vincent Favre-Nicolin@cea.fr  CEA Grenoble / INAC / SP2M
#
########################################################################
from __future__ import division
import numpy as np
import sys, os, time
from scipy import fftpack,log10,sqrt,signal,pi,arctan2,exp,linspace,newaxis,zeros,array,float32,complex64,arange,cos
from scipy.fftpack import fftn,ifftn,ifftshift, fftshift, fft2, ifft2
from pylab import gca,text,figure,imshow,title,clf,savefig,colorbar,rcParams,figtext,axes,subplot
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def complex2rgbalog(s,amin=0.5,dlogs=2):
   ph=arctan2(s.imag,s.real)
   t=pi/3
   nx,ny=s.shape
   rgba=np.zeros((nx,ny,4))
   rgba[:,:,0]=(ph<t)*(ph>-t) + (ph>t)*(ph<2*t)*(2*t-ph)/t + (ph>-2*t)*(ph<-t)*(ph+2*t)/t
   rgba[:,:,1]=(ph>t)         + (ph<-2*t)      *(-2*t-ph)/t+ (ph>0)*(ph<t)    *ph/t
   rgba[:,:,2]=(ph<-t)        + (ph>-t)*(ph<0) *(-ph)/t + (ph>2*t)         *(ph-2*t)/t
   a=np.log10(abs(s)+1e-20)
   a-=a.max()-dlogs # display dlogs orders of magnitude
   rgba[:,:,3]=amin+a/dlogs*(1-amin)*(a>0)
   return rgba

def complex2rgbalin(s):
   ph=arctan2(s.imag,s.real)
   t=pi/3
   nx,ny=s.shape
   rgba=np.zeros((nx,ny,4))
   rgba[:,:,0]=(ph<t)*(ph>-t) + (ph>t)*(ph<2*t)*(2*t-ph)/t + (ph>-2*t)*(ph<-t)*(ph+2*t)/t
   rgba[:,:,1]=(ph>t)         + (ph<-2*t)      *(-2*t-ph)/t+ (ph>0)*(ph<t)    *ph/t
   rgba[:,:,2]=(ph<-t)        + (ph>-t)*(ph<0) *(-ph)/t + (ph>2*t)         *(ph-2*t)/t
   a=np.abs(s)
   a/=a.max()
   rgba[:,:,3]=a
   return rgba

  
def getShiftLims(data):
    minx,miny=np.inf,np.inf
    maxx,maxy=-np.inf,-np.inf
    for i in data:
        if i.dx<minx:
            minx=i.dx
        if i.dy<miny:
            miny=i.dy
        if i.dx>maxx:
            maxx=i.dx
        if i.dy>maxy:
            maxy=i.dy
    return minx,maxx,miny,maxy
    
def showProgress(obj,probe,tit1='Object',tit2='Probe',figNum=100):
      from matplotlib import gridspec
      gs = gridspec.GridSpec(2, 2, height_ratios=[1, probe.shape[0]/obj.shape[0]]) 
      plt.ion()
      plt.figure(figNum)
      
      ax0 = plt.subplot(gs[0])
      ax0.imshow(np.abs(obj))
      title(tit1 + ' modulus')      

      ax1 = plt.subplot(gs[1])
      ax1.imshow(np.angle(obj))      
      title(tit1 + ' phase')            

      ax2 = plt.subplot(gs[2])
      ax2.imshow(np.abs(probe))           
      title(tit2 + ' modulus')            
      
      ax2 = plt.subplot(gs[3])      
      ax2.imshow(np.angle(probe))                           
      title(tit2 + ' phase')            
      plt.draw()
      

def showFT(p):
      a=None
      for d in p.views:
          if a==None:
              a=d.s_calc0[newaxis,:]
          else:
              a=np.concatenate((a,d.s_calc0[newaxis,:]),axis=0)          
      import plotting
      #plt.close(101)
      #plotting.imTiles(np.log(abs(a)))
      figure()
      plotting.im(np.log(abs(a[0])))
      plt.draw()        

def MakePtychoData(amplitudes,dx,dy):
    """
    Creates ptychographic dataset from the stack of amplitudes and corresponding positions.
    s_obs: amplitudes (i.e. sqrt(measured intensity of the diffraction))
    dx: vertical position
    dy: horizontal position
    """
    views=[]
    for i in arange(amplitudes.shape[0]):
       posx = dx[i]
       posy = dy[i]
       frame = amplitudes[i]
       views.append(PtychoDiffData2D(frame,posx,posy)) #o the shift posx,posy are in pixels
    return views
    
class PtychoDiffData2D:
  """
  s_obs are the amplitudes (i.e. sqrt(measured intensity of the diffraction))!
  """
  def __init__(self,s_obs,dx,dy):
    self.dx = dx # this is an amount of the shift between the different views
    self.dy = dy
    self.s_obs = s_obs.astype(float32) #o observed data

def ePIEupdate(A,psi,psi0,const):
      "Maiden2009 update for either probe or object."      
      return const*A.conjugate()*(psi-psi0)/(abs(A)**2).max()

class Ptycho2D:
  def __init__(self,views,probe0,obj0):
    self.nbproc = os.sysconf("SC_NPROCESSORS_ONLN")
    self.views = views
    self.probe = probe0.astype(complex64)
    self.probe0 = probe0.copy()
    self.obj = obj0.astype(complex64)
    self.obj0 = obj0.copy()

    try: # to do: revise this to use fftshift(fft2(fftshift(A)))
      # use FFTW
      import fftw3f # fftw for float32
      self.fftw_data = np.zeros(views[0].s_obs.shape, dtype=np.complex64)
      # create a forward and backward fft plan
      self.plan_fft = fftw3f.Plan(self.fftw_data,None, direction='forward', flags=['measure'],nthreads=4)
      self.plan_ifft = fftw3f.Plan(self.fftw_data, None, direction='backward', flags=['measure'],nthreads=4)
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
      psi0 = zeros(self.probe.shape).astype(complex64)
      ftpsi0 = zeros(self.probe.shape).astype(complex64)
      x0,y0 = self.obj.shape
      nx,ny = self.probe.shape
      cx = x0//2-nx//2+dx # This might have to be changed for objects/probes with odd number of pixels.
      cy = y0//2-ny//2+dy      
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
    """Calculate data from the object and probe. Subpixel shift.
    """
    from imProc import shift
    print "sub pix (CalcForward)"    
    if (dx!=None) and (dy!=None):
      psi0 = zeros(self.probe.shape).astype(complex64)
      ftpsi0 = zeros(self.probe.shape).astype(complex64)
      x0,y0 = self.obj.shape
      nx,ny = self.probe.shape
      cx = x0//2-nx//2
      cy = y0//2-ny//2
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
#            s_filt=signal.medfilt2d(abs(d.s_obs))
#            imFill=np.minimum(np.abs(d.s_calc0),s_filt)
            norm = (d.s_obs*mask).sum()/(np.abs(d.s_calc0)*mask).sum() # normalisation of the modulus, othervise it diverges!
#            norm=(d.s_obs[mask]).mean()/(np.abs(d.s_calc0)).mean() # normalisation of the modulus, othervise it diverges!        
            d.s_calc0_norm = d.s_calc0*norm
            imFill = np.abs(d.s_calc0_norm)
            tmp = (mask)*np.abs(d.s_obs)+(1-mask)*imFill # the missing pixels are replaced by the calculated values or by filtered values

        if DM: #Difference Map update of psi for Thibault2009
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
    x0,y0 = self.obj.shape
    nx,ny = self.probe.shape        
    if "thibault2009" in self.method.lower():
      obj = zeros(self.obj.shape).astype(complex64)
      objnorm = zeros(self.obj.shape).astype(float32)
      for d in self.views:          
        dx,dy = d.dx,d.dy #o positions of the scan
        cx = x0//2+dx-nx//2
        cy = y0//2+dy-ny//2
        objnorm[cx:cx+nx,cy:cy+ny] += abs(self.probe)**2
        obj[cx:cx+nx,cy:cy+ny] += self.probe.conjugate()*d.psi
      if "max" in self.method.lower(): #This is for Thibault2009-max
          self.obj = obj/objnorm.max()
      else:
#        self.obj = np.minimum(self.obj,1)
        reg = 1e-2*objnorm.max() #empirical threshold
        self.obj = obj/(np.maximum(objnorm,reg)) # to avoid underflow
        
    if "maiden2009" in self.method.lower():        
        a = 1;
        d = self.views[frame_index]
        dx,dy = d.dx,d.dy
        cx = x0//2+dx-nx//2
        cy = y0//2+dy-ny//2
        self.obj[cx:cx+nx,cy:cy+ny] += ePIEupdate(self.probe,d.psi,d.psi0,a)

  def UpdateObject_SubPix(self,frame_index=None):
    """
    This is equation from setting gradient of the square-of differences between the measured 
    and estimated wrt to Object to zero.  
    """
    from imProc import shift
    print "sub pix (update object)"
    x0,y0 = self.obj.shape
    nx,ny = self.probe.shape        
    if "thibault2009" in self.method.lower():
      obj = zeros(self.obj.shape).astype(complex64)
      objnorm = zeros(self.obj.shape).astype(float32)
      for d in self.views:          
        dx,dy = d.dx,d.dy #o positions of the scan
        obj_tmp = np.zeros_like(self.obj)
        cx = x0//2-nx//2
        cy = y0//2-ny//2
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

    if "maiden2009" in self.method.lower():
        a = 1;
        d = self.views[frame_index]
        dx,dy = d.dx,d.dy
        cx = x0//2+dx-nx//2
        cy = y0//2+dy-ny//2
        self.obj[cx:cx+nx,cy:cy+ny] += ePIEupdate(self.probe,d.psi,d.psi0,a)
          
    x0,y0 = self.obj.shape

    
  def UpdateProbe(self,frame_index=None):
    nx,ny = self.probe.shape
    x0,y0 = self.obj.shape    
    if "thibault2009" in self.method.lower():
      probe = zeros(self.probe.shape).astype(complex64)
      probenorm = zeros(self.probe.shape).astype(float32)
      for d in self.views:
        dx,dy = d.dx,d.dy
        cx = x0//2+dx-nx//2
        cy = y0//2+dy-ny//2
        probenorm += abs(self.obj[cx:cx+nx,cy:cy+ny])**2
        probe += self.obj[cx:cx+nx,cy:cy+ny].conjugate()*d.psi
      if "max" in self.method.lower(): #Thisis for Thibault2009-max
          self.probe = probe/probenorm.max()          
      else: 
          reg = 0.01*probenorm.max() #emprirical threshold
          self.probe = probe/np.maximum(probenorm,reg) # to avoid underflow

    if "maiden2009" in self.method.lower():
        b = 1;
        d = self.views[frame_index]
        dx,dy = d.dx,d.dy
        cx = x0//2+dx-nx//2
        cy = y0//2+dy-ny//2
        self.probe += ePIEupdate(self.obj[cx:cx+nx,cy:cy+ny],d.psi,d.psi0,b)
          
  def SaveStateOfPsiHIO(self,state=0):      
      for d in self.views:          
          initVal=np.zeros(np.append(1,d.psi0.shape),dtype=complex)
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
          d.psi0=2*d.psi0-d.psi

  def FinalUpdate(self):
      for d in self.views:
          d.psi=d.psiState[0]+d.psi-d.psiState[1]
  
  def R(self,i=None,chi2=False,mask=None,return_views=False):
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
    subplot(121)
    imshow(log10(abs(self.views[i].s_obs)),vmin=log10(abs(self.views[i].s_obs)).max()-dlogs)
    title("$Obs[%d]$"%i)
    subplot(122)
    imshow(log10(abs(self.views[i].s_calc0)),vmin=log10(abs(self.views[i].s_calc0)).max()-dlogs)
    title("$Calc[%d]$"%i)    
    
  def PlotObject(self):
    imshow(complex2rgbalin(self.obj))

  def PrintProgress(self,r, i, dt):
      print "Ptycho cycle %3d: R= %6.4f%%,  dt/cycle=%6.2fs"%(i,r*100,dt)
      s_calc_logabs = np.log10(abs(self.views[0].s_calc0).clip(1e-6))*exp(1j*np.angle(self.views[0].s_calc0))
      s_obs0 = np.log10(self.views[0].s_obs.clip(1e-6))
      showProgress(s_calc_logabs,s_obs0,tit1='s_calc0',tit2='s_obs',figNum=101)                  
      showProgress(self.obj,self.probe)

        
  def Run(self,ncycle,method="Thibault2009",updateProbe=False,verbose=False,mask=None, subPix = False):
    """
    ncycle: number of iterations
    method:
    Thibalut2009: Thibault P, Dierolf M, Bunk O, Menzel A, Pfeiffer F (2009) Probe retrieval in ptychographic coherent diffractive imaging. Ultramicroscopy 109:338-343.      
    Maiden2009: Maiden, A. M., & Rodenburg, J. M. (2009). An improved ptychographical phase retrieval algorithm for diffractive imaging. Ultramicroscopy, 109(10), 1256-62.
    updateProbe: set to True to update probe in each iteration
    verbose: set to True to print updates on the convergence, set to N>1 to plot the progres every N-steps
    mask: insert the mask of valid pixels (bad, invalid or zero pixels are zeros)
    """
    self.method = method
    print "Using method:", self.method      
    print "Evaluating %d frames."%len(self.views)
    self.mask = mask
    if self.mask!=None: print "Using mask!"
    else: print "No mask!"
    if not hasattr(self, 'Rarray'): self.Rarray=[]        

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
                  #self.UpdateObject_SubPix(method=method)
                  self.UpdateObject()
              else: 
                  self.UpdateObject()
              if updateProbe: 
                  self.UpdateProbe()
                  #self.probe=abs(self.probe0)*exp(1j*np.angle(self.probe))
                  #if ((i+1)%updateProbe)==0: 
                  #    self.UpdateProbe(method=method)
                  #else:self.UpdateProbe(method=method)
    #     self.Normalize(method="probe") # keep probe nomalied: probe.sum()=1
          self.Rarray=np.append(self.Rarray,self.R(mask=self.mask))
          if verbose:
            if i%verbose==0:
                self.PrintProgress(self.Rarray[-1],i,time.time()-t0)
    elif 'maiden' in self.method.lower():
        for i in xrange(ncycle):
            t0 = time.time()
            randSeq=np.random.permutation(arange(0,len(self.views)))
            count=0
            for j in randSeq:
                count += 1
                self.CalcForward(frame_index=j)
                self.CalcBackward(frame_index=j,mask=self.mask)
                self.UpdateObject(frame_index=j)
            if updateProbe:
                randSeq = np.random.permutation(arange(0,len(self.views)))
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
    
  def SaveResults(self, resdir="./Results/", name_appendix = ""):
        if not(os.path.isdir(resdir)): os.mkdir(resdir)  
        print "Saving results to %s<variable_name>%s.npy"%(resdir,name_appendix)
        np.save(resdir+'/obj' + name_appendix,self.obj)
        np.save(resdir+'/probe' + name_appendix,self.probe)
        np.save(resdir+'/obj0'+name_appendix,self.obj0)
        np.save(resdir+'/probe0'+name_appendix,self.probe0)
        np.save(resdir+'/R'+name_appendix,self.Rarray)        
        


"""
Simulation of the ptychographic data. Requires 
Requires imProc module:
git clone git@github.com:aludnam/imProc.git

For visualisation requires module plottools:
git clone git@github.com:aludnam/plottools.git

FZP module for simulation of the focussed beam by partially illumnated Fresnel zone plate
"""
import Image, imProc, FZP, plotting        
def get_img(index=0):    
    """
    Returns image (numpy array) from the path_list.
    
    To do: use import skimage to get images!
    """    
    path = os.path.dirname(sys.modules[__name__].__file__)
    path_list = "/Images/lena.tif", "/Images/devoogd.tif", "/Images/erika.tif"
    im = Image.open(path+path_list[index]) 
    return np.float32(np.array(im))
    
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
            plt.gca().invert_yaxis()
            plt.xlabel('pos_hor [pixels]')
            plt.ylabel('pos_ver [pixels]')
            plt.grid(b=True)
        elif np.ndim(self.values) < 3: # for 2D complex images
            plotting.showCplx(self.values)
        else:
            pass
            
class Simulation:
    def __init__(self, obj=None, obj_info={}, probe=None, probe_info={}, scan=None, scan_info={}, data_info={}, verbose = 1):
        """
        obj_info: dictionary with   obj_info['type'] = ('real','real_imag','ampl_phase') # type of the object
                                    obj_info['phase_stretch'] = ps #  specify the stretch of the phase image (default is ps=2pi)
                                    obj_info['alpha_win'] = a # ratio of the obj edge to be dimmed down by imProc.tukeywin2D. This is to artefact from sharp edges of the object. 
                                    defaults set in update_default_obj
                                    
        obj: specific value of obj can be passed (2D complex numpy array). obj_info can be passed as empty dictionary (default).
            
        probe_info: dictionary with probe_info['type'] = ('flat', 'FZP', 'Gauss',) # type of the probe
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
        self.psi = Im(np.zeros((n, s_v[0], s_v[1]), dtype = np.complex64))        
        intensity = np.zeros((n, s_a[0], s_a[1]), dtype = np.float64)
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
        
        if obj_type in ('flat','random'):
            s0,s1 = info['shape']
            im_tmp = np.zeros( (s0,s1) )
        else:
            im_tmp = get_img(0)
            s0,s1 = im_tmp.shape                                

        # Ensure the even size:        
        if s0%2: s0 -= 1
        if s1%2: s1 -= 1
        s = s0,s1
        im0 = im_tmp[:s[0],:s[1]]
            
        if obj_type == 'real':
            obj = im0        
            
        elif ('ampl' in obj_type) and ('phase' in obj_type):
            im1 = get_img(1)[:s[0],:s[1]]
            # Strethc the phase to interval (-phase_stretch/2, +phase_stretch/2)
            phase0 = im1 - im1.min()
            ps = 2*np.pi
            phase_stretch = ps*phase0/float(phase0.max()) - ps/2.        
            obj = im0 * np.exp(1j*phase_stretch)
            
        elif ('real' in obj_type) and ('imag' in obj_type):
            im1 = get_img(1)[:s[0],s[1]]
            obj = im0 + 1j*im1
            
        elif obj_type.lower() == 'random':
            rand_phase = 2*np.pi*np.random.rand(s[0],s[1])-np.pi
            obj = np.random.rand(s[0],s[1])*np.exp(1j*rand_phase)
            
        elif obj_type.lower() == 'flat':
            obj = np.ones(s).astype(complex)
            
        else:
            msg = "Unknown object type:", self.obj.info['type']
            raise NameError(msg)
            

        if 'phase_stretch' in info:         
            phase = np.angle(obj)
            phase0 = phase - phase.min()                        
            if phase0.any():                
                ps = info['phase_stretch']            
                phase_stretch = ps * phase0/float(phase0.max()) - ps/2.
                obj = abs(obj) * np.exp(1j*phase_stretch)
        
        if 'alpha_win' in info:
            w = imProc.tukeywin2D(s, info['alpha_win'])    
            obj *= w # tuckey window to smooth edges (alpha_win)
            
        self.obj.values = np.complex64(obj)

    def make_probe(self):
        """
        Simulates the beam.
        """      
        self.update_default_probe()
        info = self.probe.info
        print "Simulating the beam:",info['type']
        probe_type = info['type'].lower()
        
        if probe_type == 'flat':
            probe = np.ones(info['shape'])
            
        elif probe_type == 'gauss':
            probe = gauss2D(info['shape'], mu = (0,0), sigma = info['sigma_pix'])

        elif probe_type =='fzp': # partially illuminated Fresnel Zone Plate:
            info['gpu_name'] = "GeForce"
            info['wavelength'] = np.float32(12398.4/10030*1e-10)   # E=10 keV, April 2011 ID01
            info['focal_length'] = np.float32(.112)
            info['rmax'] = np.float32(100e-6)                       # radius of FZP
            info['nr'],info['ntheta'] = np.int32(2048),np.int32(512)         # number of points for integration on FZP (for a full illumination)
            info['r_cs'] = np.float32(40e-6)                         # Central stop radius
            info['osa_z'],info['osa_r'] = np.float32(.119),np.float32(25e-6)    # OSA position and radius            
            ps = info['shape']
            pixsize = self.data_info['pix_size_direct_nm']
            
            y = np.linspace(-pixsize*ps[0]/2,+pixsize*(ps[0]/2-1),ps[0]).astype(np.float32)
            x = y.copy()[:,np.newaxis]
            xprobe, yprobe = x*1e-9, y*1e-9 # xc,yc are in nm, x/y/zprobe are in m
            zprobe = info['focal_length']+.0003#+linspace(-.5e-3,.5e-3,256)
            xprobe = (xprobe+(yprobe+zprobe)*0).astype(np.float32)
            yprobe = (yprobe+(xprobe+zprobe)*0).astype(np.float32)
            zprobe = (zprobe+(xprobe+yprobe)*0).astype(np.float32)
            sourcex, sourcey, sourcez = np.float32(0e-6), np.float32(0e-6), np.float32(-50) # Source position (meters)            

            probe,dt,flop = FZP.FZP_thread(x=xprobe, y=yprobe, z=zprobe, sourcex=sourcex, sourcey=sourcey, sourcez=sourcez, wavelength=info['wavelength'], focal_length=info['focal_length'], rmax=info['rmax'], fzp_xmin=40e-6, fzp_xmax=60e-6, fzp_nx=256, fzp_ymin=-30e-6, fzp_ymax=30e-6, fzp_ny=256, r_cs=info['r_cs'], osa_z=info['osa_z'], osa_r=info['osa_r'], nr=info['nr'], ntheta=info['ntheta'],gpu_name=info['gpu_name'])        
            if self.verbose: 
                print "dt=%9.5fms, %8.2f Gflops"%(dt*1e3,flop/1e9/dt)           
        else: 
            msg = "Unknown probe type:", self.probe.info['type']
            raise NameError(msg)
            
        self.probe.values = np.complex64(probe)

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
            
    def show_illumination(self, log_values = False):
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