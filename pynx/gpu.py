# -*- coding: utf-8 -*-

#PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2010 Vincent Favre-Nicolin vincefn@users.sourceforge.net
#       2008-2010 Université Joseph Fourier, Grenoble (France)
#       This project was developped at the CEA/INAC/SP2M (Grenoble, France)

import numpy
import time
import os
import threading
#import multiprocessing
from scipy import weave

try:
  import pycuda.driver as drv
  import pycuda.compiler as compiler
  drv.init()
  assert drv.Device.count() >= 1
  only_cpu=False
except:
  print "PyNX: Failed importing PyCUDA, or no graphics card found => using CPU calculations only (WARNING)"
  only_cpu=True

class GPUThreads:
  threads=[]
  def __init__(self,gpu_name="GTX 295",nbCPUthread=None,verbose=False):
    self.verbose=verbose
    if self.verbose: print "Initializing GPUThreads object for: ",gpu_name
    self.gpu_name=gpu_name
    if gpu_name=="CPU" or only_cpu:
      # OSX: nbthread=int(os.popen2("sysctl -n hw.ncpu")[1].read())
      # Win: nbthread=int(os.environ["NUMBER_OF_PROCESSORS"])
      # Linux, also:nbthread=os.sysconf("SC_NPROCESSORS_CONF")
      if nbCPUthread==None: nbthread=os.sysconf("SC_NPROCESSORS_ONLN")
      else: nbthread=nbCPUthread
      for i in xrange(nbthread):
        self.threads.append(CPUThread_Fhkl())
        self.threads[-1].setDaemon(True)
        self.threads[-1].start()
    else:
      gpu_devices=[]
      for i in xrange(drv.Device.count()):
        if drv.Device(i).name().find(gpu_name)>=0:
          gpu_devices.append(i)
      nbthread=len(gpu_devices)
      for i in xrange(nbthread):
        self.threads.append(GPUThread_Fhkl(gpu_devices[i]))
        self.threads[-1].setDaemon(True)
        self.threads[-1].start()
  def __del__(self):
    if self.verbose: print "Deleting GPUThreads object"
    nbthread=len(self)
    for j in xrange(nbthread):
      self.threads[0].join_flag=True
      self.threads[0].eventStart.set()
      self.threads[0].join()
      self.threads.pop()
  def __len__(self): return len(self.threads)
  def __getitem__(self,i): return self.threads[i]
      

gputhreads=None

# 32,64,128,256 - 64 gives best performance without requiring at least 64 atoms
#TODO: check calculations - R-factor with gold gets hihger with 64 and higher values ??
#TODO:make this configurable
FHKL_BLOCKSIZE=32 

mod_fhkl_str = """
__global__ void CUDA_fhkl(float *fhkl_real,float *fhkl_imag,
                        const float *vx, const float *vy, const float *vz,const long natoms,
                        const float *vh,const float *vk,const float *vl)
{
   #define BLOCKSIZE %d
   #define twopi 6.2831853071795862f
   const unsigned long ix=threadIdx.x+blockDim.x*blockIdx.x;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __shared__ float x[BLOCKSIZE];
   __shared__ float y[BLOCKSIZE];
   __shared__ float z[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      __syncthreads();
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         float s,c;
         __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
         fr +=c;
         fi +=s;
      }
   }
   
   /* Take care of remaining atoms */
   if(threadIdx.x<(natoms-at))
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
   }
   __syncthreads();
   for(long i=0;i<(natoms-at);i++)
   {
      float s,c;
      __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
      fr +=c;
      fi +=s;
   }

   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}
"""%(FHKL_BLOCKSIZE)

mod_fhklo_str ="""
__global__ void CUDA_fhklo(float *fhkl_real,float *fhkl_imag,
                        const float *vx, const float *vy, const float *vz, const float *vocc,
                        const long natoms,
                        const float *vh,const float *vk,const float *vl)
{
   #define BLOCKSIZE %d
   #define twopi 6.2831853071795862f
   const unsigned long ix=threadIdx.x+blockDim.x*blockIdx.x;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __shared__ float x[BLOCKSIZE];
   __shared__ float y[BLOCKSIZE];
   __shared__ float z[BLOCKSIZE];
   __shared__ float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
      __syncthreads();
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         float s,c;
         __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
         fr +=occ[i]*c;
         fi +=occ[i]*s;
      }
   }
   /* Take care of remaining atoms */
   if(threadIdx.x<(natoms-at))
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
   }
   __syncthreads();
   for(long i=0;i<(natoms-at);i++)
   {
      float s,c;
      __sincosf(h*x[i] + k*y[i] + l*z[i] , &s,&c);
      fr +=occ[i]*c;
      fi +=occ[i]*s;
   }
   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}
"""%(FHKL_BLOCKSIZE)

mod_fhklo_grazing_str ="""
__global__ void CUDA_fhklo_grazing(float *fhkl_real,float *fhkl_imag,
                        const float *vx, const float *vy, const float *vz, const float *vocc,
                        const long natoms,
                        const float *vkx,const float *vky,const float *vkzr,const float *vkzi)
{
   #define BLOCKSIZE %d
   #define twopi 6.2831853071795862f
   const unsigned long ix=threadIdx.x+blockDim.x*blockIdx.x;
   const float kx=twopi*vkx[ix];
   const float ky=twopi*vky[ix];
   const float kzr=twopi*vkzr[ix];
   const float kzi=twopi*vkzi[ix];
   float fr=0,fi=0;
   __shared__ float x[BLOCKSIZE];
   __shared__ float y[BLOCKSIZE];
   __shared__ float z[BLOCKSIZE];
   __shared__ float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
      __syncthreads();
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         float s,c,atten;
         __sincosf(kx*x[i] + ky*y[i] + kzr*z[i] , &s,&c);
         atten=exp(kzi*z[i]);
         fr +=occ[i]*c*atten;
         fi +=occ[i]*s*atten;
      }
   }
   /* Take care of remaining atoms */
   if(threadIdx.x<(natoms-at))
   {
      x[threadIdx.x]=vx[at+threadIdx.x];
      y[threadIdx.x]=vy[at+threadIdx.x];
      z[threadIdx.x]=vz[at+threadIdx.x];
      occ[threadIdx.x]=vocc[at+threadIdx.x];
   }
   __syncthreads();
   for(long i=0;i<(natoms-at);i++)
   {
      float s,c,atten;
      __sincosf(kx*x[i] + ky*y[i] + kzr*z[i] , &s,&c);
      atten=exp(kzi*z[i]);
      fr +=occ[i]*c*atten;
      fi +=occ[i]*s*atten;
   }
   fhkl_real[ix]+=fr;
   fhkl_imag[ix]+=fi;
}
"""%(FHKL_BLOCKSIZE)

code_CPU_fhkl_xyz="""
  const float PI2         = 6.28318530717958647692528676655900577f;
  for(unsigned long i=0;i<nhkl;i++)
  {
      float fr=0,fi=0;
      const float h=vh[i]*PI2;
      const float k=vk[i]*PI2;
      const float l=vl[i]*PI2;
      const float * __restrict__ px=vx;
      const float * __restrict__ py=vy;
      const float * __restrict__ pz=vz;
      __m128 vfr,vfi,vs,vc,vtmp;
      float tmp[4];
      for(unsigned long at=0;at<natoms;at+=4)
      {
        float * __restrict__ ptmp=&tmp[0];
        
        // Dangerous ? Order of operation is not guaranteed - but it works...
        sincos_ps(_mm_set_ps(h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++,
                             h* *px++ +k * *py++ + l * *pz++),&vs,&vc);
        if(at==0) 
        {vfr=vc;vfi=vs;}
        else 
        {vfr=_mm_add_ps(vfr,vc);vfi=_mm_add_ps(vfi,vs);}
      }
      float tmp2[4];
      _mm_store_ps(tmp2,vfr);
      for(unsigned int j=0;j<4;++j) fr+=tmp2[j];
      _mm_store_ps(tmp2,vfi);
      for(unsigned int j=0;j<4;++j) fi+=tmp2[j];
      freal[i]=fr;
      fimag[i]=fi;
  }
"""

# "Fast" (SSE-optimized) calculation using CPU -requires sse_mathfun.h
class CPUThread_Fhkl(threading.Thread):
  """ (internal)
  Fast (SSE-optimized) calculation using CPU -requires sse_mathfun.h
  """
  def __init__(self, verbose=False):
    threading.Thread.__init__(self)
    """ Here we assume that the number of atoms is a multiple of 4
    0-padding must already have been done by the calling program
    """
    self.verbose=verbose
    self.dt=0.0
    self.eventStart=threading.Event()
    self.eventFinished=threading.Event()
    self.join_flag=False
  def run(self):
    if self.verbose: print self.name," ...beginning"
    while True:
      self.eventStart.wait()
      if self.join_flag: break
      if self.verbose: print self.name," ...got a job !"
      
      t0=time.time()
      nhkl  =len(self.h.flat)
      natoms=len(self.x.flat)
      
      self.fhkl_real=numpy.empty(self.h.shape,dtype=numpy.float32)
      self.fhkl_imag=numpy.empty(self.h.shape,dtype=numpy.float32)
      freal=self.fhkl_real
      fimag=self.fhkl_imag
      vh,vk,vl=self.h,self.k,self.l
      vx,vy,vz=self.x,self.y,self.z
      
      # TODO: take into account occupancy !
      pth= os.path.dirname(os.path.abspath(__file__))
      fhkl_xyz = weave.inline("Py_BEGIN_ALLOW_THREADS\n" + code_CPU_fhkl_xyz+ "Py_END_ALLOW_THREADS\n",
                              ['freal','fimag','vh', 'vk', 'vl', 'vx', 'vy', 'vz', 'nhkl', 'natoms'],
                              extra_compile_args=["-O3 -I"+pth+" -w -ffast-math -msse -msse2  -msse3 -msse4.1 -march=core2 -mfpmath=sse -fstrict-aliasing -pipe -fomit-frame-pointer -funroll-loops -ftree-vectorize -ftree-vectorizer-verbose=0"],
                              compiler = 'gcc',
                              support_code="""#define USE_SSE2
                                              #include "sse_mathfun.h"
                                            """,
                              include_dirs=['./'])
      
      self.dt=time.time()-t0
      self.eventStart.clear()
      self.eventFinished.set()
    #MRAtS=nhkl*float(natoms)/self.dt/1e6

class GPUThread_Fhkl(threading.Thread):
  """(internal)
  Class to compute Fhkl in a single thread
  """
  def __init__(self, devID,verbose=False):
    threading.Thread.__init__(self)
    """ Here we assume that the number of hkl is a multiple of 32
    0-padding must already have been done by the calling program
    """
    assert drv.Device.count() >= devID+1
    self.devID = devID
    self.verbose=verbose
    self.dt=0.0
    if self.verbose:print drv.Device(self.devID).name()
    self.eventStart=threading.Event()
    self.eventFinished=threading.Event()
    self.join_flag=False
  def run(self):
    dev = drv.Device(self.devID)
    ctx = dev.make_context()
    
    # Kernel will be initialized when necessary
    CUDA_fhkl,CUDA_fhk,CUDA_fhklo,CUDA_fhko,CUDA_fhkl_grazing=None,None,None,None,None
    
    BLOCKSIZE           =dev.get_attribute(drv.device_attribute.WARP_SIZE) # 32
    MULTIPROCESSOR_COUNT=dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)
    if self.verbose: print self.name," ...beginning"
    while True:
      self.eventStart.wait()
      if self.join_flag: break
      if self.verbose: print self.name," ...got a job !"
      
      t0=time.time()
      nhkl0  =numpy.int32(len(self.h.flat))
      natoms=numpy.uint32(len(self.x.flat))
      
      nhkl=len(self.h)
      
      self.fhkl_real=0*self.h
      self.fhkl_imag=0*self.h
      
      step_atoms=2L**19
      if long(natoms)*nhkl>(2**31L*MULTIPROCESSOR_COUNT):
        # To avoid spending more than 5s per call
        step_atoms=long(2**19L*2**31L*MULTIPROCESSOR_COUNT/BLOCKSIZE/(long(natoms)*nhkl))
        step_atoms*=BLOCKSIZE
      steps_nbatoms=range(0,natoms,step_atoms)
         
      if steps_nbatoms[-1]!=natoms: steps_nbatoms.append(natoms)
      
      # "The maximum size of each dimension of a grid of thread blocks is 65535"
      steps_nhkl=range(0,nhkl,65535*FHKL_BLOCKSIZE)
      if steps_nhkl[-1]!=nhkl: steps_nhkl.append(nhkl)
      
      #if self.verbose: print "Atom ranges:",steps_nbatoms
      for j in xrange(1,len(steps_nhkl)):# not always optimal, separate in equal sizes would be better
        for i in xrange(1,len(steps_nbatoms)):# not always optimal, separate in equal sizes would be better
          tmpx=self.x[steps_nbatoms[i-1]:steps_nbatoms[i]]
          tmpy=self.y[steps_nbatoms[i-1]:steps_nbatoms[i]]
          tmpz=self.z[steps_nbatoms[i-1]:steps_nbatoms[i]]
          tmpocc=None
          if self.occ!=None:
            tmpocc=self.occ[steps_nbatoms[i-1]:steps_nbatoms[i]]
          #if self.verbose: print [steps_nbatoms[i-1],steps_nbatoms[i]]
          if type(self.occ)==type(None) and type(self.vkzi)==type(None):
            if CUDA_fhkl==None:
              if self.verbose: print "Compiling CUDA_fhkl"
              mod_fhkl = compiler.SourceModule(mod_fhkl_str, options=["-use_fast_math"])
              CUDA_fhkl = mod_fhkl.get_function("CUDA_fhkl")
            CUDA_fhkl (drv.InOut(self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.InOut(self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]]), 
                      drv.In(tmpx),
                      drv.In(tmpy),
                      drv.In(tmpz),numpy.int32(len(tmpx)),
                      drv.In(self.h[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.k[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.l[steps_nhkl[j-1]:steps_nhkl[j]]),block=(FHKL_BLOCKSIZE,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//FHKL_BLOCKSIZE,1))
          if type(self.occ)!=type(None) and type(self.vkzi)==type(None):
            if CUDA_fhklo==None:
              if self.verbose: print "Compiling CUDA_fhklo"
              mod_fhklo = compiler.SourceModule(mod_fhklo_str, options=["-use_fast_math"])
              CUDA_fhklo = mod_fhklo.get_function("CUDA_fhklo")
                
            CUDA_fhklo(drv.InOut(self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.InOut(self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]]), 
                      drv.In(tmpx),
                      drv.In(tmpy),
                      drv.In(tmpz),
                      drv.In(tmpocc),numpy.int32(len(tmpx)),
                      drv.In(self.h[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.k[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.l[steps_nhkl[j-1]:steps_nhkl[j]]),block=(FHKL_BLOCKSIZE,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//FHKL_BLOCKSIZE,1))
          if type(self.occ)!=type(None) and type(self.vkzi)!=type(None):
            if CUDA_fhkl_grazing==None:
              if self.verbose:print "Compiling CUDA_fhklo_grazing"
              mod_fhkl_grazing = compiler.SourceModule(mod_fhklo_grazing_str, options=["-use_fast_math"])
              CUDA_fhkl_grazing = mod_fhkl_grazing.get_function("CUDA_fhklo_grazing")
            CUDA_fhkl_grazing (drv.InOut(self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.InOut(self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]]), 
                      drv.In(tmpx),
                      drv.In(tmpy),
                      drv.In(tmpz),
                      drv.In(tmpocc),
                      numpy.int32(len(tmpx)),
                      drv.In(self.h   [steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.k   [steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.l   [steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]]),block=(FHKL_BLOCKSIZE,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//FHKL_BLOCKSIZE,1))
          if type(self.occ)==type(None) and type(self.vkzi)!=type(None):
            #Also use CUDA_fhklo_grazing, just create a temporary occ=1, performance loss is negligeable
            if CUDA_fhkl_grazing==None:
              if self.verbose:print "Compiling CUDA_fhklo_grazing"
              mod_fhkl_grazing = compiler.SourceModule(mod_fhklo_grazing_str, options=["-use_fast_math"])
              CUDA_fhkl_grazing = mod_fhkl_grazing.get_function("CUDA_fhklo_grazing")
            
            CUDA_fhkl_grazing (drv.InOut(self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.InOut(self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]]), 
                      drv.In(tmpx),
                      drv.In(tmpy),
                      drv.In(tmpz),
                      drv.In(numpy.ones(tmpz.shape).astype(numpy.float32)),
                      numpy.int32(len(tmpx)),
                      drv.In(self.h   [steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.k   [steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.l   [steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]]),block=(FHKL_BLOCKSIZE,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//FHKL_BLOCKSIZE,1))
         
        self.dt=time.time()-t0
        self.eventStart.clear()
        self.eventFinished.set()
    #MRAtS=nhkl*float(natoms)/self.dt/1e6
    ctx.pop()

def Fhkl_thread(h,k,l,x,y,z,occ=None,verbose=False,gpu_name="GTX 295",nbCPUthread=None,sz_imag=None):
   """
   Compute          F(hkl)=SUM_i exp(2j*pi*(h*x_i  + k*y_i  + l*z_i))
   or equivalently: F(k)  =SUM_i exp(2j*pi*(sx*x_i + sy*y_i + sz*z_i))
   
   nbCPUthread can be used only for CPU computing, when no GPU is available. nbCPUthread can
   be set to the number of cores available. Using None (the default) makes the program recognize
   the number of available processors/cores
   
   If sz_imag is not equal to None, then it means that the scattering vector s has an imaginary
   component (absorption), e.g. because of grazing incidence condition.
   """
   global gputhreads
   if gputhreads==None:
      gputhreads=GPUThreads(gpu_name,nbCPUthread=nbCPUthread,verbose=verbose)
   elif gputhreads.gpu_name!=gpu_name:
     # GPU has changed, re-initialize
     gputhreads=None
     gputhreads=GPUThreads(gpu_name,nbCPUthread=nbCPUthread,verbose=verbose)
   # Make sure (h,k,l) and (x,y,z) all have the same size,
   # force pseudo-1d to 3D, and change type to float32
   if h.shape==k.shape and h.shape==l.shape and h.dtype==numpy.float32 and k.dtype==numpy.float32 and l.dtype==numpy.float32:
    vh=h.ravel()
    vk=k.ravel()
    vl=l.ravel()
   else:
    vh=(h+(k+l)*0).astype(numpy.float32)
    vk=(k+(h+l)*0).astype(numpy.float32)
    vl=(l+(h+k)*0).astype(numpy.float32)
    vh=vh.reshape(len(vh.flat))
    vk=vk.reshape(len(vh.flat))
    vl=vl.reshape(len(vh.flat))

   if type(sz_imag)!=type(None):
     if sz_imag.shape==vh.shape and sz_imag.dtype==numpy.float32:
       vkzi=sz_imag.ravel()
     else:
       vkzi=((h+k+l)*0+sz_imag).astype(numpy.float32)
       vkzi=vkzi.reshape(len(vh.flat))
   else:
     vkzi=None
   
   nhkl0  =numpy.int32(len(vh))
   nhkl=nhkl0
   
   # We need a multiple of 32 hkl (WARPSIZE)
   WARPSIZE=32
   d,m=divmod(nhkl0,WARPSIZE)
   nhkl=nhkl0
   if m!=0:
      if verbose: print "nhkl=%d is not a multiple of %d, using 0-padding"%(nhkl0,WARPSIZE)
      nhkl=numpy.int32((d+1)*WARPSIZE)
      vh=numpy.resize(vh,nhkl)
      vk=numpy.resize(vk,nhkl)
      vl=numpy.resize(vl,nhkl)
      vkzi=numpy.resize(vkzi,nhkl)
   
   # Force float32 type
   if x.shape==y.shape and x.shape==z.shape and x.dtype==numpy.float32 and y.dtype==numpy.float32 and z.dtype==numpy.float32:
    vx=x.ravel()
    vy=y.ravel()
    vz=z.ravel()
   else:
    vx=(x+(y+z)*0).astype(numpy.float32)
    vy=(y+(x+z)*0).astype(numpy.float32)
    vz=(z+(x+y)*0).astype(numpy.float32)
   
   natoms=numpy.uint32(len(vx.flat))
   vx.resize(natoms)
   vy.resize(natoms)
   vz.resize(natoms)
   
   vocc=None
   if occ!=None:
      vocc=numpy.resize(occ+(x+y+z)*0,natoms)
      vocc=(vocc+vx*0).astype(numpy.float32)
   
   # Create as many threads as available devices
   t0=time.time()
   nbthread=len(gputhreads)
   #threads=[]
   for i in xrange(nbthread):
      a0=i*int(natoms/nbthread)
      a1=(i+1)*int(natoms/nbthread)
      if verbose: print "Thread #",i,[a0,a1]
      if i==(nbthread-1): a1=natoms
      tmpocc=vocc
      if occ!=None: tmpocc=vocc[a0:a1]
      gputhreads[i].verbose=verbose
      gputhreads[i].h=vh
      gputhreads[i].k=vk
      gputhreads[i].l=vl
      gputhreads[i].vkzi=vkzi
      gputhreads[i].x=vx[a0:a1]
      gputhreads[i].y=vy[a0:a1]
      gputhreads[i].z=vz[a0:a1]
      gputhreads[i].occ=tmpocc
      gputhreads[i].eventFinished.clear()
      gputhreads[i].eventStart.set()
   for i in xrange(nbthread):
      gputhreads[i].eventFinished.wait()

      #gpu_thread = GPUThread_Fhkl(gpu_devices[i],vh,vk,vl,vx[a0:a1],vy[a0:a1],vz[a0:a1],tmpocc,verbose)
      #gpu_thread.start()
      #threads.append(gpu_thread)
      
   #for t in threads:t.join()
   fhkl=(0+0j)*vh
   for i in xrange(nbthread): 
      t=gputhreads[i]
      fhkl += t.fhkl_real+1j*t.fhkl_imag
      if verbose: print "Thread #%d, dt=%7.5f"%(i,t.dt)
   #for i in xrange(nbthread): del threads[0]
   dt=time.time()-t0
   MRAtS=nhkl*float(natoms)/dt/1e6*nbthread
   if nhkl!=nhkl0:# get back to original hkl size
      fhkl= numpy.resize(fhkl,nhkl0)
   return fhkl.reshape((h+k+l).shape),dt

def Fhkl_gold(h,k,l,x,y,z,occ=None,verbose=False,dtype=numpy.float64):
   """
   Compute reference value (using CPU) of:
      F(hkl)=SUM_i exp(2j*pi*(h*x_i + k*y_i + l*z_i))
   """
   vh=(h+(k+l)*0).astype(dtype)
   vk=(k+(h+l)*0).astype(dtype)
   vl=(l+(h+k)*0).astype(dtype)
   nhkl0  =len(vh.flat)
   
   vx=(x+(y+z)*0).astype(dtype)
   vy=(y+(x+z)*0).astype(dtype)
   vz=(z+(x+y)*0).astype(dtype)
   natoms=len(vx.flat)
   
   fhkl=0j*vh
   t0=time.time()
   if type(occ)==type(None):
         occ=numpy.float32(1.0)
   for i in xrange(len(vh.flat)):
      fhkl.flat[i]=(occ*numpy.exp(2j*numpy.pi*(vh.flat[i]*vx+vk.flat[i]*vy+vl.flat[i]*vz))).sum()
   return fhkl,time.time()-t0
