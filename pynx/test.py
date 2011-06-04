# -*- coding: utf-8 -*-
from numpy import arange,sin,exp,pi,newaxis,float32,random,ones
from pynx import gpu

def test_fhkl(gpu_name,nx=40,ny=40,nz=40,nh=40,nk=40,nl=40,verbose=False):
  #Create array of 3D coordinates, 50x50x50 cells
  x=arange(0,nx,dtype=float32)
  y=arange(0,ny,dtype=float32)[:,newaxis]
  z=arange(0,nz,dtype=float32)[:,newaxis,newaxis]
  #HKL coordinates as a 2D array
  h=random.uniform(.01,.5,nh)
  k=random.uniform(.01,.5,nk)[:,newaxis]
  l=random.uniform(2.01,2.5,nl)[:,newaxis,newaxis]
  #The actual computation (done twice - first time to compile kernel)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,gpu_name=gpu_name,verbose=verbose)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,gpu_name=gpu_name,verbose=verbose)
  #Compare to analytical formula
  fhkl_gold=(exp(1j*pi*h*(nx-1)) * sin(pi*h*nx) / sin(pi*h))  *  (exp(1j*pi*k*(ny-1)) * sin(pi*k*ny) / sin(pi*k))  *  (exp(1j*pi*l*(nz-1)) * sin(pi*l*nz) / sin(pi*l))
  diff=abs(fhkl-fhkl_gold)
  tmp=diff.mean()/abs(fhkl_gold).mean()
  s="PASS"
  if tmp>0.01:s="FAIL"
  print "%10s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%6.3f 10^9 reflections.atoms/s  =>   <|GPU-analytical|> / <|analytical|>=%7.5f, %s"%("fhkl",nh*nk*nl//1000,nx*ny*nz//1000,nx*ny*nz*nh*nk*nl/dt/1e9, tmp,s)
  return fhkl,fhkl_gold

def test_fhklo(gpu_name,nx=40,ny=40,nz=40,nh=40,nk=40,nl=40,verbose=False):
  #Create array of 3D coordinates, 50x50x50 cells
  x=arange(0,nx,dtype=float32)
  y=arange(0,ny,dtype=float32)[:,newaxis]
  z=arange(0,nz,dtype=float32)[:,newaxis,newaxis]
  occ=ones((x+y+z).shape)*0.4
  #HKL coordinates as a 2D array
  h=random.uniform(.01,.5,nh)
  k=random.uniform(.01,.5,nk)[:,newaxis]
  l=random.uniform(2.01,2.5,nl)[:,newaxis,newaxis]
  #The actual computation (done twice - first time to compile kernel)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,occ,gpu_name=gpu_name,verbose=verbose)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,occ,gpu_name=gpu_name,verbose=verbose)
  #Compare to analytical formula
  fhkl_gold=occ.mean()*(exp(1j*pi*h*(nx-1)) * sin(pi*h*nx) / sin(pi*h))  *  (exp(1j*pi*k*(ny-1)) * sin(pi*k*ny) / sin(pi*k))  *  (exp(1j*pi*l*(nz-1)) * sin(pi*l*nz) / sin(pi*l))
  diff=abs(fhkl-fhkl_gold)
  tmp=diff.mean()/abs(fhkl_gold).mean()
  s="PASS"
  if tmp>0.01:s="FAIL"
  print "%10s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%6.3f 10^9 reflections.atoms/s  =>   <|GPU-analytical|> / <|analytical|>=%7.5f, %s"%("fhklo",nh*nk*nl//1000,nx*ny*nz//1000,nx*ny*nz*nh*nk*nl/dt/1e9, tmp,s)
  return fhkl,fhkl_gold

def test_fhklo_graz(gpu_name,nx=40,ny=40,nz=40,nh=40,nk=40,nl=40,verbose=False):
  #Create array of 3D coordinates, 50x50x50 cells
  x=arange(0,nx,dtype=float32)
  y=arange(0,ny,dtype=float32)[:,newaxis]
  z=-arange(0,nz,dtype=float32)[:,newaxis,newaxis] #z negative, atoms below the surface 
  occ=ones((x+y+z).shape)*0.4
  #HKL coordinates as a 2D array
  h=random.uniform(.01,.5,nh)
  k=random.uniform(.01,.5,nk)[:,newaxis]
  l=random.uniform(2.01,2.5,nl)[:,newaxis,newaxis]
  kz_imag=ones(l.shape)*0.01
  #The actual computation (done twice - first time to compile kernel)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,occ,gpu_name=gpu_name,kz_imag=kz_imag,verbose=verbose)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,occ,gpu_name=gpu_name,kz_imag=kz_imag,verbose=verbose)
  #Compare to analytical formula
  fhkl_gold=occ.mean()*(exp(1j*pi*h*(nx-1)) * sin(pi*h*nx) / sin(pi*h))  *  (exp(1j*pi*k*(ny-1)) * sin(pi*k*ny) / sin(pi*k))  *  ( exp(-2*pi*(1j*l+kz_imag)*nz)-1 ) / ( exp(-2*pi*(1j*l+kz_imag))-1 )
  diff=abs(fhkl-fhkl_gold)
  tmp=diff.mean()/abs(fhkl_gold).mean()
  s="PASS"
  if tmp>0.01:s="FAIL"
  print "%10s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%6.3f 10^9 reflections.atoms/s  =>   <|GPU-analytical|> / <|analytical|>=%7.5f, %s"%("fhklo_graz",nh*nk*nl//1000,nx*ny*nz//1000,nx*ny*nz*nh*nk*nl/dt/1e9, tmp,s)
  return fhkl,fhkl_gold

def test_fhkl_graz(gpu_name,nx=40,ny=40,nz=40,nh=40,nk=40,nl=40,verbose=False):
  #Create array of 3D coordinates, 50x50x50 cells
  x=arange(0,nx,dtype=float32)
  y=arange(0,ny,dtype=float32)[:,newaxis]
  z=-arange(0,nz,dtype=float32)[:,newaxis,newaxis] #z negative, atoms below the surface 
  #HKL coordinates as a 2D array
  h=random.uniform(.01,.5,nh)
  k=random.uniform(.01,.5,nk)[:,newaxis]
  l=random.uniform(2.01,2.5,nl)[:,newaxis,newaxis]
  kz_imag=ones(l.shape)*0.01
  #The actual computation (done twice - first time to compile kernel)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,gpu_name=gpu_name,kz_imag=kz_imag,verbose=verbose)
  fhkl,dt=gpu.Fhkl_thread(h,k,l,x,y,z,gpu_name=gpu_name,kz_imag=kz_imag,verbose=verbose)
  #Compare to analytical formula
  fhkl_gold=(exp(1j*pi*h*(nx-1)) * sin(pi*h*nx) / sin(pi*h))  *  (exp(1j*pi*k*(ny-1)) * sin(pi*k*ny) / sin(pi*k))  *  ( exp(-2*pi*(1j*l+kz_imag)*nz)-1 ) / ( exp(-2*pi*(1j*l+kz_imag))-1 )
  diff=abs(fhkl-fhkl_gold)
  tmp=diff.mean()/abs(fhkl_gold).mean()
  s="PASS"
  if tmp>0.01:s="FAIL"
  print "%10s: %5d 10^3 reflections, %5d 10^3 atoms, speed=%6.3f 10^9 reflections.atoms/s  =>   <|GPU-analytical|> / <|analytical|>=%7.5f, %s"%("fhkl_graz",nh*nk*nl//1000,nx*ny*nz//1000,nx*ny*nz*nh*nk*nl/dt/1e9, tmp,s)
  return fhkl,fhkl_gold

def test_all(nx=40,ny=40,nz=40,nh=40,nk=40,nl=40,verbose=False):
  for i in xrange(gpu.drv.Device.count()):
    print "######## PyNX: testing for device: ",gpu.drv.Device(i).name()
    junk=test_fhkl      (gpu.drv.Device(i).name(),nx,ny,nz,nh,nk,nl,verbose=verbose)
    junk=test_fhklo     (gpu.drv.Device(i).name(),nx,ny,nz,nh,nk,nl,verbose=verbose)
    junk=test_fhklo_graz(gpu.drv.Device(i).name(),nx,ny,nz,nh,nk,nl,verbose=verbose)
    junk=test_fhkl_graz (gpu.drv.Device(i).name(),nx,ny,nz,nh,nk,nl,verbose=verbose)


