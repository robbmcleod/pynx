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
  import pyopencl as cl
except:pass

try:
  import pycuda.driver as drv
  import pycuda.compiler as compiler
except:pass


class GPUThreads:
  threads=[]
  def __init__(self,gpu_name="GTX 295",language="OpenCL",cl_platform="",nbCPUthread=None,verbose=False):
    self.verbose=True
    if self.verbose: print "Initializing GPUThreads object for: "+gpu_name+" (language="+language+","+cl_platform+")"
    self.gpu_name=gpu_name
    self.language=language
    self.cl_platform=cl_platform
    self.cl_devices=[]
    self.cuda_devices=[]
    if language.lower()=="opencl":
      try:
        tmp=[]
        for p in cl.get_platforms():
          if p.name.find(cl_platform)>=0:
            tmp+=p.get_devices()
        for d in tmp:
          if d.name.find(gpu_name)>=0:
            self.cl_devices.append(d)
        nbthread=len(self.cl_devices)
        for i in xrange(nbthread):
          self.threads.append(OpenCLThread_Fhkl(self.cl_devices[i],verbose=verbose))
          self.threads[-1].setDaemon(True)
          self.threads[-1].start()
      except:
        print "PyNX: Failed importing PyOpenCL, or no platform/graphics card (paltform="+cl_platform+", gpu_name="+gpu_name+") found => using CPU calculations only (WARNING)"
    elif language.lower()=="cuda":
      try:
        drv.init()
        cuda_devices=[]
        for i in xrange(drv.Device.count()):
          if drv.Device(i).name().find(gpu_name)>=0:
            self.cuda_devices.append(i)
        nbthread=len(self.cuda_devices)
        for i in xrange(nbthread):
          self.threads.append(CUDAThread_Fhkl(self.cuda_devices[i],verbose=verbose))
          self.threads[-1].setDaemon(True)
          self.threads[-1].start()
      except:
        print "PyNX: Failed importing PyCUDA, or no graphics card (gpu_name="+gpu_name+") found => using CPU calculations only (WARNING)"
    
    if len(self.cuda_devices)==0 and len(self.cl_devices)==0:
      # OSX: nbthread=int(os.popen2("sysctl -n hw.ncpu")[1].read())
      # Win: nbthread=int(os.environ["NUMBER_OF_PROCESSORS"])
      # Linux, also:nbthread=os.sysconf("SC_NPROCESSORS_CONF")
      if nbCPUthread==None: nbthread=os.sysconf("SC_NPROCESSORS_ONLN")
      else: nbthread=nbCPUthread
      for i in xrange(nbthread):
        self.threads.append(CPUThread_Fhkl(verbose=verbose))
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

###########################################################################################################
###########################################################################################################
#####################################        CUDA KERNELS        ##########################################
###########################################################################################################
###########################################################################################################

# 32,64,128,256 - 64 gives best performance without requiring at least 64 atoms
#TODO: check calculations - R-factor with gold gets hihger with 64 and higher values ??
#TODO:make this configurable
#FHKL_BLOCKSIZE=32 

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
      __syncthreads();
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
   __syncthreads();
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
"""

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
      __syncthreads();
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
   __syncthreads();
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
"""

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
      __syncthreads();
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
   __syncthreads();
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
"""

###########################################################################################################
###########################################################################################################
#####################################          CPU CODE          ##########################################
###########################################################################################################
###########################################################################################################


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

###########################################################################################################
###########################################################################################################
#####################################        OpenCL Kernels      ##########################################
###########################################################################################################
###########################################################################################################
CL_FHKL_CODE="""

__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz, const long natoms,
                   __global float *vh,__global float *vk,__global float *vl)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi 6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=h*x[i] + k*y[i] + l*z[i];
         fi +=sin(tmp);
         fr +=cos(tmp);
      }
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=h*x[i] + k*y[i] + l*z[i];
      fi +=sin(tmp);
      fr +=cos(tmp);
   }

   fhkl_real[ix]=fr;
   fhkl_imag[ix]=fi;
}"""

CL_FHKLO_CODE="""
__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
__kernel void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz,__global float *vocc, const long natoms,
                   __global float *vh,__global float *vk,__global float *vl)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi 6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float h=twopi*vh[ix];
   const float k=twopi*vk[ix];
   const float l=twopi*vl[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   __local float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      occ[tx]=vocc[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=h*x[i] + k*y[i] + l*z[i];
         fi +=occ[i]*sin(tmp);
         fr +=occ[i]*cos(tmp);
      }
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=h*x[i] + k*y[i] + l*z[i];
      fi +=occ[i]*sin(tmp);
      fr +=occ[i]*cos(tmp);
   }

   fhkl_real[ix]=fr;
   fhkl_imag[ix]=fi;
}"""

CL_FHKL_grazing_CODE="""

__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz, const long natoms,
                   __global float *vkx,__global float *vky,__global float *vkzr,__global float *vkzi)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi 6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float kx=twopi*vkx[ix];
   const float ky=twopi*vky[ix];
   const float kzr=twopi*vkzr[ix];
   const float kzi=twopi*vkzi[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
         const float atten=exp(kzi*z[i]);
         fi +=sin(tmp)*atten;
         fr +=cos(tmp)*atten;
      }
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
      const float atten=exp(kzi*z[i]);
      fi +=sin(tmp)*atten;
      fr +=cos(tmp)*atten;
   }

   fhkl_real[ix]=fr;
   fhkl_imag[ix]=fi;
}"""

CL_FHKLO_grazing_CODE="""
__kernel __attribute__((reqd_work_group_size(%(block_size)d, 1, 1))) 
__kernel void Fhkl(__global float *fhkl_real,__global float *fhkl_imag,
                   __global float *vx,__global float *vy,__global float *vz,__global float *vocc, const long natoms,
                   __global float *vkx,__global float *vky,__global float *vkzr,__global float *vkzi)
{
   #define BLOCKSIZE %(block_size)d
   #define twopi 6.2831853071795862f
   // Block index
   int bx = get_group_id(0);
   int by = get_group_id(1);

   // Thread index
   int tx = get_local_id(0);
   //int ty = get_local_id(1);
    
   const unsigned long ix=tx+(bx+by*get_num_groups(0))*BLOCKSIZE;
   const float kx=twopi*vkx[ix];
   const float ky=twopi*vky[ix];
   const float kzr=twopi*vkzr[ix];
   const float kzi=twopi*vkzi[ix];
   float fr=0,fi=0;
   __local float x[BLOCKSIZE];
   __local float y[BLOCKSIZE];
   __local float z[BLOCKSIZE];
   __local float occ[BLOCKSIZE];
   long at=0;
   for (;at<=(natoms-BLOCKSIZE);at+=BLOCKSIZE)
   {
      barrier(CLK_LOCAL_MEM_FENCE);
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
      occ[tx]=vocc[at+tx];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(unsigned int i=0;i<BLOCKSIZE;i++)
      {
         // it is faster to calc separately sin and cos,
         // since native_ versions of thoses exist.
         //
         // in CUDA arch, the sin and cos will actually be computed
         // together (http://forums.nvidia.com/index.php?s=ea1305aede92c332301f1d0fe6a53237&showtopic=30069&view=findpost&p=169542)
         const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
         const float atten=exp(kzi*z[i]);
         fi +=occ[i]*sin(tmp)*atten;
         fr +=occ[i]*cos(tmp)*atten;
      }
   }
   /* Take care of remaining atoms */
   if(tx<(natoms-at))
   {
      x[tx]=vx[at+tx];
      y[tx]=vy[at+tx];
      z[tx]=vz[at+tx];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(long i=0;i<(natoms-at);i++)
   {
      const float tmp=kx*x[i] + ky*y[i] + kzr*z[i];
      const float atten=exp(kzi*z[i]);
      fi +=occ[i]*sin(tmp)*atten;
      fr +=occ[i]*cos(tmp)*atten;
   }

   fhkl_real[ix]=fr;
   fhkl_imag[ix]=fi;
}"""

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
    self.block_size=32     #would work with just 4 ?
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

class CUDAThread_Fhkl(threading.Thread):
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
    
    self.block_size=64#           =dev.get_attribute(drv.device_attribute.WARP_SIZE) # 32
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
        step_atoms=long(2**19L*2**31L*MULTIPROCESSOR_COUNT/self.block_size/(long(natoms)*nhkl))
        step_atoms*=self.block_size
      steps_nbatoms=range(0,natoms,step_atoms)
         
      if steps_nbatoms[-1]!=natoms: steps_nbatoms.append(natoms)
      
      # "The maximum size of each dimension of a grid of thread blocks is 65535"
      steps_nhkl=range(0,nhkl,65535*self.block_size)
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
              if self.verbose: print "Compiling CUDA_fhkl (block size=%d)"%self.block_size
              mod_fhkl = compiler.SourceModule(mod_fhkl_str%(self.block_size), options=["-use_fast_math"])
              CUDA_fhkl = mod_fhkl.get_function("CUDA_fhkl")
            CUDA_fhkl (drv.InOut(self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.InOut(self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]]), 
                      drv.In(tmpx),
                      drv.In(tmpy),
                      drv.In(tmpz),numpy.int32(len(tmpx)),
                      drv.In(self.h[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.k[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.l[steps_nhkl[j-1]:steps_nhkl[j]]),block=(self.block_size,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//self.block_size,1))
          if type(self.occ)!=type(None) and type(self.vkzi)==type(None):
            if CUDA_fhklo==None:
              if self.verbose: print "Compiling CUDA_fhklo (block size=%d)"%self.block_size
              mod_fhklo = compiler.SourceModule(mod_fhklo_str%(self.block_size), options=["-use_fast_math"])
              CUDA_fhklo = mod_fhklo.get_function("CUDA_fhklo")
                
            CUDA_fhklo(drv.InOut(self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.InOut(self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]]), 
                      drv.In(tmpx),
                      drv.In(tmpy),
                      drv.In(tmpz),
                      drv.In(tmpocc),numpy.int32(len(tmpx)),
                      drv.In(self.h[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.k[steps_nhkl[j-1]:steps_nhkl[j]]),
                      drv.In(self.l[steps_nhkl[j-1]:steps_nhkl[j]]),block=(self.block_size,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//self.block_size,1))
          if type(self.occ)!=type(None) and type(self.vkzi)!=type(None):
            if CUDA_fhkl_grazing==None:
              if self.verbose:print "Compiling CUDA_fhklo_grazing (block size=%d)"%self.block_size
              mod_fhkl_grazing = compiler.SourceModule(mod_fhklo_grazing_str%(self.block_size), options=["-use_fast_math"])
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
                      drv.In(self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]]),block=(self.block_size,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//self.block_size,1))
          if type(self.occ)==type(None) and type(self.vkzi)!=type(None):
            #Also use CUDA_fhklo_grazing, just create a temporary occ=1, performance loss is negligeable
            if CUDA_fhkl_grazing==None:
              if self.verbose:print "Compiling CUDA_fhklo_grazing (block size=%d)"%self.block_size
              mod_fhkl_grazing = compiler.SourceModule(mod_fhklo_grazing_str%(self.block_size), options=["-use_fast_math"])
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
                      drv.In(self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]]),block=(self.block_size,1,1),grid=((steps_nhkl[j]-steps_nhkl[j-1])//self.block_size,1))
         
        self.dt=time.time()-t0
        self.eventStart.clear()
        self.eventFinished.set()
    #MRAtS=nhkl*float(natoms)/self.dt/1e6
    ctx.pop()

class OpenCLThread_Fhkl(threading.Thread):
  """(internal)
  Class to compute Fhkl in a single thread (single OpenCL platform/device)
  """
  def __init__(self, dev,verbose=False):
    threading.Thread.__init__(self)
    """ Here we assume that the number of hkl is a multiple of 32
    0-padding must already have been done by the calling program
    """
    self.dev = dev
    self.verbose=verbose
    self.dt=0.0
    if self.verbose:print self.dev.name
    self.eventStart=threading.Event()
    self.eventFinished=threading.Event()
    self.join_flag=False
  def run(self):
    ctx = cl.Context([self.dev])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    # Kernel program will be initialized when necessary
    CL_fhkl,CL_fhklo,CL_fhkl_grazing,CL_fhklo_grazing=None,None,None,None
    
    self.block_size=128 #good default for NVidia/AMD cards ?
    if self.dev.max_work_group_size<self.block_size:
      self.block_size=self.dev.max_work_group_size
    MULTIPROCESSOR_COUNT=self.dev.max_compute_units
    if self.verbose: print self.name," ...beginning"
    
    kernel_params={"block_size":self.block_size}

    #if "NVIDIA" in queue.device.vendor:
    options = "-cl-mad-enable -cl-fast-relaxed-math" # -cl-unsafe-math-optimizations

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
        # To avoid spending more than 5s per call (CUDA threshold - may need to be adapted for other platforms)
        step_atoms=long(2**19L*2**31L*MULTIPROCESSOR_COUNT/self.block_size/(long(natoms)*nhkl))
        step_atoms*=self.block_size
      steps_nbatoms=range(0,natoms,step_atoms)
         
      if steps_nbatoms[-1]!=natoms: steps_nbatoms.append(natoms)
      
      steps_nhkl=range(0,nhkl,65535*self.block_size)
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
            if CL_fhkl==None:
              if self.verbose: print "Compiling CL_fhkl (block size=%d)"%self.block_size
              CL_fhkl = cl.Program(ctx, CL_FHKL_CODE % kernel_params,).build(options=options)
            h_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h[steps_nhkl[j-1]:steps_nhkl[j]], size=self.h[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            k_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.k[steps_nhkl[j-1]:steps_nhkl[j]], size=self.k[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            l_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.l[steps_nhkl[j-1]:steps_nhkl[j]], size=self.l[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)

            x_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpx, size=tmpx.nbytes)
            y_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpy, size=tmpy.nbytes)
            z_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpz, size=tmpz.nbytes)

            fhkl_real_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            fhkl_imag_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            CL_fhkl.Fhkl(queue, (steps_nhkl[j]-steps_nhkl[j-1], 1), (self.block_size,1), fhkl_real_, fhkl_imag_, x_, y_, z_, numpy.int64(len(tmpx)), h_, k_, l_)
            cl.enqueue_copy(queue, self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_real_).wait()
            cl.enqueue_copy(queue, self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_imag_).wait()
            
          if type(self.occ)!=type(None) and type(self.vkzi)==type(None):
            if CL_fhklo==None:
              if self.verbose: print "Compiling CL_fhklo (block size=%d)"%self.block_size
              CL_fhklo = cl.Program(ctx, CL_FHKLO_CODE % kernel_params,).build(options=options)
            h_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h[steps_nhkl[j-1]:steps_nhkl[j]], size=self.h[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            k_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.k[steps_nhkl[j-1]:steps_nhkl[j]], size=self.k[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            l_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.l[steps_nhkl[j-1]:steps_nhkl[j]], size=self.l[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)

            x_   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpx  , size=tmpx.nbytes)
            y_   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpy  , size=tmpy.nbytes)
            z_   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpz  , size=tmpz.nbytes)
            occ_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpocc, size=tmpz.nbytes)

            fhkl_real_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            fhkl_imag_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            
            CL_fhklo.Fhkl(queue, (steps_nhkl[j]-steps_nhkl[j-1], 1), (self.block_size,1), fhkl_real_, fhkl_imag_, x_, y_, z_, occ_, numpy.int64(len(tmpx)), h_, k_, l_)
            cl.enqueue_copy(queue, self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_real_).wait()
            cl.enqueue_copy(queue, self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_imag_).wait()
            
          if type(self.occ)!=type(None) and type(self.vkzi)!=type(None):
            if CL_fhklo_grazing==None:
              if self.verbose: print "Compiling CL_fhklo_grazing (block size=%d)"%self.block_size
              CL_fhklo_grazing = cl.Program(ctx, CL_FHKLO_grazing_CODE % kernel_params,).build(options=options)
            h_    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h   [steps_nhkl[j-1]:steps_nhkl[j]], size=self.h   [steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            k_    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.k   [steps_nhkl[j-1]:steps_nhkl[j]], size=self.k   [steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            l_    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.l   [steps_nhkl[j-1]:steps_nhkl[j]], size=self.l   [steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            vkzi_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]], size=self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)

            x_   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpx  , size=tmpx.nbytes)
            y_   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpy  , size=tmpy.nbytes)
            z_   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpz  , size=tmpz.nbytes)
            occ_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpocc, size=tmpocc.nbytes)

            fhkl_real_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            fhkl_imag_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            
            CL_fhklo_grazing.Fhkl(queue, (steps_nhkl[j]-steps_nhkl[j-1], 1), (self.block_size,1), fhkl_real_, fhkl_imag_, x_, y_, z_, occ_, numpy.int64(len(tmpx)), h_, k_, l_,vkzi_)
            cl.enqueue_copy(queue, self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_real_).wait()
            cl.enqueue_copy(queue, self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_imag_).wait()

          if type(self.occ)==type(None) and type(self.vkzi)!=type(None):
            if CL_fhkl_grazing==None:
              if self.verbose: print "Compiling CL_fhkl_grazing (block size=%d)"%self.block_size
              CL_fhkl_grazing = cl.Program(ctx, CL_FHKL_grazing_CODE % kernel_params,).build(options=options)
            h_    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h   [steps_nhkl[j-1]:steps_nhkl[j]], size=self.h   [steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            k_    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.k   [steps_nhkl[j-1]:steps_nhkl[j]], size=self.k   [steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            l_    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.l   [steps_nhkl[j-1]:steps_nhkl[j]], size=self.l   [steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            vkzi_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]], size=self.vkzi[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)

            x_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpx, size=tmpx.nbytes)
            y_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpy, size=tmpy.nbytes)
            z_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=tmpz, size=tmpz.nbytes)

            fhkl_real_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            fhkl_imag_ = cl.Buffer(ctx, mf.WRITE_ONLY, size=self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]].nbytes)
            
            CL_fhkl_grazing.Fhkl(queue, (steps_nhkl[j]-steps_nhkl[j-1], 1), (self.block_size,1), fhkl_real_, fhkl_imag_, x_, y_, z_, numpy.int64(len(tmpx)), h_, k_, l_,vkzi_)
            cl.enqueue_copy(queue, self.fhkl_real[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_real_).wait()
            cl.enqueue_copy(queue, self.fhkl_imag[steps_nhkl[j-1]:steps_nhkl[j]], fhkl_imag_).wait()
         
        self.dt=time.time()-t0
        self.eventStart.clear()
        self.eventFinished.set()

def Fhkl_thread(h,k,l,x,y,z,occ=None,verbose=False,gpu_name="GTX 295",nbCPUthread=None,sz_imag=None,language="OpenCL",cl_platform=""):
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
     if verbose: print "Fhkl_thread: initilizing gputhreads with GPU=%s, language=%s,cl_platform=%s"%(gpu_name,language,cl_platform)
     gputhreads=GPUThreads(gpu_name,nbCPUthread=nbCPUthread,verbose=verbose,language=language,cl_platform=cl_platform)
   elif gputhreads.gpu_name!=gpu_name or gputhreads.language!=language or gputhreads.cl_platform!=cl_platform:
     # GPU has changed, re-initialize
     gputhreads=None
     if verbose: print "Fhkl_thread: initilizing gputhreads with GPU=%s, language=%s,cl_platform=%s"%(gpu_name,language,cl_platform)
     gputhreads=GPUThreads(gpu_name,nbCPUthread=nbCPUthread,verbose=verbose,language=language,cl_platform=cl_platform)
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
   
   # We need a multiple of WARPSIZE hkl
   BLOCKSIZE=1
   for t in gputhreads.threads:
     if t.block_size>BLOCKSIZE:
       BLOCKSIZE=t.block_size
   d,m=divmod(nhkl0,BLOCKSIZE)
   nhkl=nhkl0
   if m!=0:
      if verbose: print "nhkl=%d is not a multiple of %d, using 0-padding"%(nhkl0,BLOCKSIZE)
      nhkl=numpy.int32((d+1)*BLOCKSIZE)
      vh=numpy.resize(vh,nhkl)
      vk=numpy.resize(vk,nhkl)
      vl=numpy.resize(vl,nhkl)
      if vkzi!=None:vkzi=numpy.resize(vkzi,nhkl)
   
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

      #gpu_thread = CUDAThread_Fhkl(gpu_devices[i],vh,vk,vl,vx[a0:a1],vy[a0:a1],vz[a0:a1],tmpocc,verbose)
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
