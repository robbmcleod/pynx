# -*- coding: utf-8 -*-

#PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2010 Vincent Favre-Nicolin vincefn@users.sourceforge.net
#       2008-2010 Université Joseph Fourier, Grenoble (France)
#       This project was developped at the CEA/INAC/SP2M (Grenoble, France)

import scipy
from scipy import pi

import cctbx
from cctbx import crystal
from cctbx import xray
from cctbx import miller
from cctbx import uctbx
from cctbx import sgtbx
from cctbx.array_family import flex
from cctbx.eltbx import sasaki
from cctbx.eltbx import tiny_pse

#Electron classical radius (Angstroems)
re=2.814e-15

try:
  from pynx import gpu
  has_pynx_gpu=True
except:
  has_pynx_gpu=False

def W2E(x):
  """ nrj->wavelength or wavelength->nrj , nrj in eV, wavelength in Angstroems"""
  return 12398.4/x

class Scatterer (xray.scatterer):
  """ Energy in eV"""              
  # Is this class really necessary ? Only for f' f" storage..
  def __init__(self,label,site,occup,u,energy,fp=None,fs=None):
    xray.scatterer.__init__(self,label=label,site=site,occupancy=occup,u=u)
    self.nrj=energy
    if fp==None or fs==None:
      fp_fdp_sasaki=sasaki.table(self.element_symbol()).at_ev(self.nrj)
      self.fp0=fp_fdp_sasaki.fp()
      self.fs0=fp_fdp_sasaki.fdp()
    else:
      self.fp0=fp0
      self.fs0=fs
  def GetF0(self,stol):
    scatt=flex.xray_scatterer((self,))
    scatt_dict=xray.ext.scattering_dictionary(scatt)
    scatt_dict.assign_from_table("WK1995")
    return scatt_dict.dict()[self.label].gaussian.at_stol(stol)

class Crystal:
  def __init__(self,unitcell,spacegroup,scatterers):
    self.sg=sgtbx.space_group_info(spacegroup)
    self.uc=uctbx.unit_cell(unitcell)
    self.scatt=scatterers
    self.cctbx_scatterers=flex.xray_scatterer()
    for s in self.scatt:
      self.cctbx_scatterers.append(xray.scatterer(label=s.label,site=s.site,occupancy=s.occupancy,u=s.u_iso))
    xray.add_scatterers_ext(unit_cell=self.uc,
                            space_group=self.sg.group(),
                            scatterers=self.cctbx_scatterers,
                            site_symmetry_table=sgtbx.site_symmetry_table(),
                            site_symmetry_table_for_new=sgtbx.site_symmetry_table(),
                            min_distance_sym_equiv=0.5,
                            u_star_tolerance=0,
                            assert_min_distance_sym_equiv=True)
    
    cs=crystal.symmetry(self.uc,spacegroup)
    sp=crystal.special_position_settings(cs)
    self.structure=xray.structure(sp,self.cctbx_scatterers)
    self.structure_as_P1=self.structure.expand_to_p1()
  def GetF0(self,scatterer,h,k,l):
    scatt=flex.xray_scatterer((scatterer,))
    scatt_dict=xray.ext.scattering_dictionary(scatt)
    scatt_dict.assign_from_table("WK1995")
    return scatt_dict.dict()[scatt.label].gaussian.at_stol( self.uc.stol((h,k,l)) )
  def CalcMultiplicity(self):
    for i in self.scatt:
        ss = i.apply_symmetry(self.uc, self.sg.group())
  def GetAtomDensity(self):
    """ Number of atoms per cubic meter"""
    vol=self.uc.volume()*1e-30
    self.CalcMultiplicity()
    density=dict()
    for s in self.scatt:
      #site symmetry      #TODO : calculate real multiplicity for special positions !
      #ssym=s.apply_symmetry(self.uc,self.sg.group())
      density[s.label]=s.occupancy*s.multiplicity()/vol
    return density
  def GetDensity(self):
    """ kg/m^3 """
    vol=self.uc.volume()*1e-30
    self.CalcMultiplicity()
    density=0
    for s in self.scatt:
      #site symmetry      #TODO : calculate real multiplicity for special positions !
      #ssym=s.apply_symmetry(self.uc,self.spg.group())
      tpse=tiny_pse.table(s.element_symbol())
      density+=s.occupancy*s.multiplicity()*tpse.weight()/1000/(vol*6.0221353e23)
    return density
  def GetRefractionIndexDeltaBeta(self):
    """ Refraction index, n=1-delta-i*beta. returns a tuple delta,beta"""
    delta=0.0
    beta=0.0
    density=self.GetAtomDensity()
    wav=W2E(self.scatt[0].nrj)*1e-10 #assume all scatterers have same list of nrjs
    density=self.GetAtomDensity()
    for s in self.scatt:
      z = sasaki.table(s.label).atomic_number()
      fp_fdp_sasaki=sasaki.table(s.element_symbol()).at_ev(self.scatt[0].nrj)
      delta+=wav**2*density[s.label]*re/(2*pi)*(z-s.fp0)
      beta +=wav**2*density[s.label]*re/(2*pi)*s.fs0
    return delta,beta
  def GetCriticalAngle(self):
    delta,beta=self.GetRefractionIndexDeltaBeta()
    return scipy.sqrt(2*delta)
  def GetLinearAbsorptionCoeff(self):
    mu=0.0
    density=self.GetAtomDensity()
    for s in self.scatt:
      mu+=5639.9e-28*W2E(s.nrj)*s.fs0*density[s.label]
    return mu

class Wave:
  def __init__(self,alphai,eparall,eperp,nrj):
    """
        Incident wave above a surface.
        Coordinates:
          - z is perpendicular to the surface, >0 going UP (different from H Dosch's convention)
          - x is the projection of the wavevector on the surface
          - y is parallel to the surface
        
        alphai: incident angle, with respect to the surface
        eparallel: component of the electric field parallel to the incident plane (vertical plane)
        eperp: component of the electric field perpendicular to the incident plane (along y)
        nrj: values of the energy of the incident wave, in eV
        
        alphai *or* nrj can be arrays, but not together
    """
    self.alphai=alphai
    self.eparall=eparall
    self.eperp=eperp
    self.ex=scipy.sin(alphai)*eparall
    self.ey=eperp
    self.ez=scipy.cos(alphai)*eparall
    self.kx= 2*pi/W2E(nrj)*scipy.cos(alphai)
    self.ky= 2*pi/W2E(nrj)*0
    self.kz=-2*pi/W2E(nrj)*scipy.sin(alphai)
    self.nrj=nrj

class DistortedWave:
  """
    Reflected and refracted wave at a surface
  """
  def __init__(self,material0,material1,wave):
    if material0==None:
        self.delta,self.beta=material1.GetRefractionIndexDeltaBeta()
    else:
        delta0,beta0=material0.GetRefractionIndexDeltaBeta()
        delta1,beta1=material1.GetRefractionIndexDeltaBeta()
        self.delta=delta1-delta0
        self.beta =beta1 -beta0
    n=1-self.delta-self.beta*1j
    self.sinr_alphai=(scipy.sin(wave.alphai)**2-2*self.delta-self.beta*2j)**0.5
    self.krx= wave.kx*1.0
    self.kry= wave.ky*0
    self.krz=-wave.kz
    self.ktx=wave.kx
    self.kty=0.0
    self.ktz=-2*pi/W2E(wave.nrj)*self.sinr_alphai
    self.Tix=2*self.sinr_alphai /(n**2*scipy.sin(wave.alphai)+self.sinr_alphai)
    self.Tiy=2*scipy.sin(wave.alphai)/(     scipy.sin(wave.alphai)+self.sinr_alphai)
    self.Tiz=2*scipy.sin(wave.alphai)/(n**2*scipy.sin(wave.alphai)+self.sinr_alphai)
    self.Rix=-(n**2*scipy.sin(wave.alphai)-self.sinr_alphai)/(n**2*scipy.sin(wave.alphai)+self.sinr_alphai)
    self.Riy= (     scipy.sin(wave.alphai)-self.sinr_alphai)/(     scipy.sin(wave.alphai)+self.sinr_alphai)
    self.Riz= (n**2*scipy.sin(wave.alphai)-self.sinr_alphai)/(n**2*scipy.sin(wave.alphai)+self.sinr_alphai)
    self.PenetrationDepth=1/self.ktz.imag # in Angstroems
    self.erx=self.Rix*wave.ex
    self.ery=self.Riy*wave.ey
    self.erz=self.Riz*wave.ez
    self.etx=self.Tix*wave.ex
    self.ety=self.Tiy*wave.ey
    self.etz=self.Tiz*wave.ez
    ne2=abs(wave.ex)**2+abs(wave.ey)**2+abs(wave.ez)**2
    net2=abs(self.etx)**2+abs(self.ety)**2+abs(self.etz)**2
    ner2=abs(self.erx)**2+abs(self.ery)**2+abs(self.erz)**2
    self.Ti2=net2/ne2
    self.Ri2=ner2/ne2

def FhklDWBA4(x,y,z,h,k,l=None,occ=None,alphai=0.2,alphaf=None,substrate=None,wavelength=1.0,e_par=0.,e_perp=1.0,gpu_name="CPU"):
  """
  Calculate the grazing-incidence X-ray scattered intensity taking into account
  4 scattering paths, for a nanostructure object located above a given substrate.
  
  x,y,z: coordinates of the atoms in fractionnal coordinates (relative to the 
  substrate unit cell)
  h,k,l: reciprocal space coordinates
  alphai, alphaf: incident and outgoing angles, in radians
  wavelength: in Angstroems
  e_par,e_perp: percentage of polarisation parallel and perpendicular to the incident plane
  
  Note: Either l *OR* alphaf must be supplied - it is assumed that the lattice
  coordinates are such that the [001] direction is perpendicular to the surface.
  """
  nrj=W2E(wavelength)
  c=substrate.uc.parameters()[2]
  if alphaf==None:
    alphaf=scipy.arcsin(l/2/c*wavelength)

  # Incident wave
  w=Wave(alphai,0,1.0,nrj)
  dw=DistortedWave(None,substrate,w)
  # Reflected wave after the dot
  w1=Wave(alphaf,0,1.0,nrj)
  dw1=DistortedWave(None,substrate,w1)

  # First path, direct diffraction
  l=c*2*scipy.sin(alphaf+alphai)/wavelength
  f1=gpu.Fhkl_thread(h,k,l,x,y,z,occ=occ,gpu_name=gpu_name)[0]

  # Second path, reflection before
  l=c*2*scipy.sin(alphaf-alphai)/wavelength
  f2=gpu.Fhkl_thread(h,k,l,x,y,z,occ=occ,gpu_name=gpu_name)[0]*dw.Riy

  # Third path, reflection after dot
  l=c*2*scipy.sin(-alphaf+alphai)/wavelength
  f3=gpu.Fhkl_thread(h,k,l,x,y,z,occ=occ,gpu_name=gpu_name)[0]*dw1.Riy

  # Fourth path, reflection before and after dot
  l=c*2*scipy.sin(-alphaf-alphai)/wavelength
  f4=gpu.Fhkl_thread(h,k,l,x,y,z,occ=occ,gpu_name=gpu_name)[0]*dw.Riy*dw1.Riy
  return f1+f2+f3+f4
