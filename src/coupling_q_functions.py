import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time
from scipy import interpolate
import pandas as pd
import h5py
import pdb
import cmath
from numpy import fft
from tqdm import tqdm

class shape:
	'''trapezoids and squares only'''
	def __init__(self,origin,ang,w,h,λ,material):
		self.materials_list = {
			'aln':self.n_aln,
			'sio2':self.n_sio2,
			'al2o3':self.n_al2o3,
			'linbo3_z':self.n_linbo3_z
		}
		assert(len(origin)==2)
		self.o = origin
		self.v1,self.v2,self.v3 = self.find_vertices(origin,ang,w,h)
		if material not in self.materials_list:
			raise ValueError('material not defined, consider adding it into the shape class')
		else:
			self.index = self.materials_list[material](λ)
		
	def find_vertices(self,origin,ang,w,h):
		v1 = [origin[0]+h*ang,origin[1]+h]
		v2 = [v1[0]+w,v1[1]]
		v3 = [origin[0]+2*h*ang+w,origin[1]]
		return v1,v2,v3
		
	def n_aln(self,λ):
		return np.sqrt( 1 + 3.318*λ**2/( λ**2 - 0.138**2 ) + 
					 3.71*λ**2/( λ**2 - 16.4**2 ) )
	
	def n_sio2(self,λ):
		return np.sqrt( 1 + 0.6961663*λ**2/(λ**2 - 0.0684043**2) + 
					 0.4079426*λ**2/(λ**2 - 0.1162414**2) +
					 0.8974794*λ**2/(λ**2 - 9.896161**2) )
	
	def n_al2o3(self,λ):
		return np.sqrt( 1 + 1.4313493*λ**2/(λ**2-0.0726631**2) + 
					 0.65054713*λ**2/(λ**2-0.1193242**2) + 
					 5.3414021*λ**2/(λ**2-18.028251**2) )

	def n_linbo3_z(self,λ):
		return np.sqrt( 1.0 + 2.9804*λ**2/(λ**2-0.02047) + 0.5981*λ**2/(λ**2-0.0666) + 8.9543*λ**2/(λ**2-416.08) )
		
def get_efield(f,res,mode,mode_n,N,section):
	assert (mode=='TM') or (mode=='TE')
	f.Exec(f"{section}.evlist.update")
	te_count = 0; tm_count = 0
	found = 0
	for i in range(N):
		mode_i = i+1
		tefrac = f.Exec(f"{section}.evlist.list[{mode_i}].modedata.tefrac")
		if mode == 'TE':
			if (tefrac > 50) and (te_count == mode_n):
				f.Exec(f"{section}.evlist.list[{mode_i}].profile.update")
				found = 1
				break
			elif tefrac <= 50:
				tm_count += 1
			else:
				te_count += 1
		elif mode == 'TM':
			if (tefrac <= 50) and (tm_count == mode_n):
				f.Exec(f"{section}.evlist.list[{mode_i}].profile.update")
				found = 1
				break
			elif tefrac > 50:
				te_count += 1
			else:
				tm_count += 1
	if found == 0:
		raise ValueError('mode was not found, a larger value of nmodes is needed')
	# allocate resources
	Ex = np.zeros((res+1,res+1))
	Ey = np.zeros((res+1,res+1))
	Ez = np.zeros((res+1,res+1))
	# Ex
	f.Exec(f"Set Ax={section}.evlist.list[{mode_i}].profile.data.getfieldarray(1)")
	E = f.Exec(f"Ax.fieldarray")
	E = list(map(lambda x: eval(x)[0],E.split()[1::2]))
	assert(len(E) == ((res+1)*(res+1)))
	idx = 0
	for x in range(res+1):
		for y in range(res+1):
			Ex[x][y] += E[idx]
			idx += 1
	
	# Ey
	f.Exec(f"Set Ay={section}.evlist.list[{mode_i}].profile.data.getfieldarray(2)")
	E = f.Exec(f"Ay.fieldarray")
	E = list(map(lambda x: eval(x)[0],E.split()[1::2]))
	assert(len(E) == ((res+1)*(res+1)))
	idx = 0
	for x in range(res+1):
		for y in range(res+1):
			Ey[x][y] += E[idx]
			idx += 1
	
	# Ez
	f.Exec(f"Set Az={section}.evlist.list[{mode_i}].profile.data.getfieldarray(3)")
	E = f.Exec(f"Az.fieldarray")
	E = list(map(lambda x: eval(x)[0],E.split()[1::2]))
	assert(len(E) == ((res+1)*(res+1)))
	idx = 0
	for x in range(res+1):
		for y in range(res+1):
			Ey[x][y] += E[idx]
			idx += 1
	
	return Ex,Ey,Ez,mode_i

def get_neff(f,section,mode_i):
	f.Exec(f"{section}.evlist.update")
	return np.real(f.Exec(f"{section}.evlist.list[{mode_i}].neff"))

def get_index(shapes,x_size,y_size,res):
	''' in order to make this function more general, it will always take a list
	of shapes (classes).  Every shape will have the following attributes:

	o: specifies the point of origin of your shape (ALWAYS BOTTOM LEFT, change
	in FIMMWAVE if necessary), should be updated everytime there is a change in 
	the geometry
	v1: next vertex which should be following the clockwise direction
	v2: next vertex
	v3: next vertex
	index: bulk material index of the shape

	order the shapes such that each successive shape overrides the previous one
	
	*** ASSUMPTION ***
	y boundaries of the shapes are horizontal (slope of 0)
	'''
	index_arr = np.ones((res+1,res+1))
	
	for shape in shapes:
		for ix in range(res+1):
			for iy in range(res+1):
				x = ix*x_size/res
				y = iy*y_size/res
				left_line_slope = (shape.v1[0]-shape.o[0])/(shape.v1[1]-shape.o[1])
				right_line_slope = (shape.v2[0]-shape.v3[0])/(shape.v2[1]-shape.v3[1])
				left_line_pt = left_line_slope*(y-shape.o[1])+shape.o[0]
				right_line_pt = right_line_slope*(y-shape.v3[1])+shape.v3[0]
				if (x >= left_line_pt) and (x <= right_line_pt) and (y >= shape.o[1]) and (y <= shape.v1[1]):
					index_arr[ix][iy] = shape.index
				else:
					pass

	return index_arr

def get_ng(f,index):
	f.Exec(f"section_ring.evlist.list[{index}].modedata.update(1)")
	ng = f.Exec(f"section_ring.evlist.list[{index}].modedata.neffg")
	return ng

def get_overlap(f,ex_ring,ey_ring,ez_ring,ex_bus,ey_bus,ez_bus,mode_i_bus,
				ring_index_arr,bus_index_arr,ring_shape,bus_shape,
				radius,gap,rw,bw,res,x_size,y_size,guess_ng=False):
	# find mode volume of ring and mode area of bus and normalization factors
	# refer to 6 February 2006 / Vol. 14, No. 3 / OPTICS EXPRESS 1105 for mode volume calculation
	# refer to pg. 106 of M. Soltani, “Novel integrated silicon nanophotonic structures using ultra-high Q resonators,” 
	# Ph.D. thesis,Georgia Institute of Technology (2009). for normalization
	ϵ0 = 8.85e-12 # F/m
	c = 299792458
	um = 1e-6
	r_arr = (np.ones((301,301))*np.arange(0,x_size+x_size/res,x_size/res)+radius-x_size/2).T
	r_arr_ring = r_arr*np.ones((res+1,res+1))*(ring_index_arr == ring_shape.index)
	r_arr_bus = r_arr*np.ones((res+1,res+1))*(bus_index_arr == bus_shape.index)
	mode_volume_ring = 2*np.pi*np.trapz(np.trapz((np.abs(ex_ring)**2+(np.abs(ey_ring)**2)/(np.amax(np.abs(ex_ring))**2+np.amax(np.abs(ey_ring))**2))*r_arr_ring*um,
										 dx=x_size*um/res,axis=0),
								dx=y_size*um/res)
	mode_area_bus = 1*np.trapz(np.trapz(((np.abs(ex_bus)**2+(np.abs(ey_bus)**2))/(np.amax(np.abs(ex_bus))**2+np.amax(np.abs(ey_bus))**2)),
										 dx=x_size*um/res,axis=0),
								dx=y_size*um/res)
	U = 0.5*ϵ0*ring_shape.index**2*(np.abs(np.amax(np.abs(ex_ring)))**2+np.abs(np.amax(np.abs(ey_ring)))**2)*mode_volume_ring
	if guess_ng:
		P = 0.5*ϵ0*bus_shape.index**2*(np.abs(np.amax(np.abs(ex_bus)))**2+np.abs(np.amax(np.abs(ey_bus)))**2)*mode_area_bus*c/guess_ng
	else:
		P = 0.5*ϵ0*bus_shape.index**2*(np.abs(np.amax(np.abs(ex_bus)))**2+np.abs(np.amax(np.abs(ey_bus)))**2)*mode_area_bus*c/get_ng(f,mode_i_bus)
	# calculate normalized field overlap
	area_to_int = np.ones((res+1,res+1))*(bus_index_arr == bus_shape.index)
	integrand = ((bus_index_arr**2-ring_index_arr**2)*area_to_int*
	             (ex_ring*ex_bus+ey_ring*ey_bus)*r_arr_bus*um/(np.sqrt(U)*np.sqrt(P)))
	overlap = ϵ0*np.trapz(np.trapz(integrand,dx=x_size*um/res,axis=0),
					   dx=y_size*um/res)
	return overlap,r_arr_bus,r_arr_ring

def get_Qc(overlap,λ,neff_ring,neff_bus,radius,gap,rw,bw,wraparound_ang):
	ϵ0 = 8.85e-12 # F/m
	c = 299792458
	um = 1e-6
	ω = 2*np.pi*c/(λ*um)
	k0 = 2*np.pi/(λ*um) # free-space wavevector
	ang = (wraparound_ang*np.pi/180)/2
	ring_r = radius*um
	bus_r = (radius+rw/2+bw/2+gap)*um
	const = (2/(k0*(neff_bus*bus_r-neff_ring*ring_r)))
	# refer to February 2010 / Vol. 18, No. 3 / OPTICS EXPRESS 2130 for Qc calculation
	# also good reference: IEEE JOURNAL OF QUANTUM ELECTRONICS, VOL. 35, NO. 9, SEPTEMBER 1999
	κ = (1j*ω*overlap/4)*const*np.sin(ang/(2*const))
	Qc = ω/np.abs(κ)**2
	return Qc