# this code calculates the coupling q of a bus-waveguide/ring resonator system
from pdPythonLib import *
# All functions and classes used by this coupling Q calculation are imported in the following line
from coupling_q_functions import *

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
import argparse
import matplotlib

# user input variables
# useful to set min_te and max_te sometimes 
ring_mode = 'TM'; bus_mode = 'TM'
ring_mode_n = 1; bus_mode_n = 0
Nmodes = 10
field_resolution = 300
x_size = 9; y_size = 4
sidewall_angle = 10
ang = np.tan(sidewall_angle*np.pi/180)
gap = 0.45
radius = 100
height = 1
offset_x = 1.5
wraparound_ang = 20 # in degrees
wrap_ang = wraparound_ang*2*np.pi/360

# vary parameters
ring_w = [4.5]
bus_w = [1]#np.linspace(0.8,1.1,21) # [0.5,0.55,0.6,0.65]
pump_λ = np.linspace(1.500,1.600,21) #[1.550] #np.linspace(1.2,1.8,51)#

def coupling_q(ring_mode,bus_mode,ring_mode_n,bus_mode_n,Nmodes,field_resolution,x_size,y_size,
				sidewall_angle,gap,radius,height,offset_x,wraparound_ang,ring_w,bus_w,pump_λ,
				waveguide_material,substrate_material,cladding_material):
	# set up environment
	f = pdApp()
	f.ConnectToApp()
	print('Successfully connected to FIMMWAVE')
	ang = np.tan(sidewall_angle*np.pi/180)
	wrap_ang = wraparound_ang*2*np.pi/360
	section_ring = 'Ref& section_ring = app.findnode("/coupling_q/ring")'
	section_bus = 'Ref& section_bus = app.findnode("/coupling_q/bus")'
	variables = 'Ref& variables = app.findnode("/coupling_q/variables")'
	f.Exec(section_ring)
	f.Exec(section_bus)
	f.Exec(variables)
	print('Succesfully assigned geometry and variables')
	# physical constants
	c = 299792458
	ħ = 6.634e-34/2/np.pi

	print('wrap-around angle =',wraparound_ang)
	print('wrap-around length =',wraparound_ang*2*np.pi*radius/360)

	f.Exec(f"variables.setvariable(N,{Nmodes})")
	f.Exec(f"variables.setvariable(resolution,{field_resolution})")
	f.Exec(f"variables.setvariable(a,{x_size})")
	f.Exec(f"variables.setvariable(b,{y_size})")
	f.Exec(f"variables.setvariable(ang,{ang})")
	f.Exec(f"variables.setvariable(extract_gap,{gap})")
	f.Exec(f"variables.setvariable(radius,{radius})")
	f.Exec(f"variables.setvariable(h,{height})")
	f.Exec(f"variables.setvariable(offset_x,{offset_x})")

	progress = tqdm(total=len(ring_w)*len(bus_w)*len(pump_λ))
	Qc = np.zeros((len(ring_w),len(bus_w),len(pump_λ)))
	for i,rw in enumerate(ring_w):
		f.Exec(f"variables.setvariable(w,{rw})")
		for j,bw in enumerate(bus_w):
			f.Exec(f"variables.setvariable(sin_top_w,{bw})")
			for k,λ in enumerate(pump_λ):
				f.Exec(f"variables.setvariable(lambda_ir,{λ})")
				section = 'section_ring'
				radius_offset_ring = radius-(height*ang+rw/2)+height*ang+gap/2+rw-offset_x
				f.Exec(f"variables.setvariable(radius,{radius_offset_ring})")
				# define shapes
				sio2_shape = shape(origin=[0,0],ang=0,w=x_size,
								   h=y_size,λ=λ,material=cladding_material)
				al2o3_shape = shape(origin=[0,0],ang=0,w=x_size,
								   h=y_size/2-height/2,λ=λ,material=substrate_material)
				ring_shape = shape(origin=[x_size/2-rw-(height*ang)-gap/2+offset_x,
										   y_size/2-height/2],
								   ang=ang,w=rw,h=height,λ=λ,material=waveguide_material)
				bus_shape = shape(origin=[x_size/2-(height*ang)+gap/2+offset_x,
										   y_size/2-height/2],
								   ang=ang,w=bw,h=height,λ=λ,material=waveguide_material)
				shapes_ring = [sio2_shape,al2o3_shape,ring_shape]
				shapes_bus = [sio2_shape,al2o3_shape,bus_shape]
				
				# solve coupling q
				# ring part
				ex_ring,ey_ring,ez_ring,mode_i_ring = get_efield(f,field_resolution,
																 ring_mode,ring_mode_n,
																 Nmodes,section)
				neff_ring = get_neff(f,section,mode_i_ring)
				ring_index_arr = get_index(shapes_ring,x_size,y_size,field_resolution)
				
				# bus waveguide part
				section = 'section_bus'
				ex_bus,ey_bus,ez_bus,mode_i_bus = get_efield(f,field_resolution,
															 bus_mode,bus_mode_n,
															 Nmodes,section)
				neff_bus = get_neff(f,section,mode_i_bus)
				print('ring mode =',mode_i_ring,'bus mode =',mode_i_bus)
				bus_index_arr = get_index(shapes_bus,x_size,y_size,field_resolution)
				overlap,ring_arr,bus_arr = get_overlap(f,ex_ring,ey_ring,ez_ring,ex_bus,ey_bus,ez_bus,mode_i_bus,
									ring_index_arr,bus_index_arr,ring_shape,bus_shape,
									radius_offset_ring,gap,rw,bw,field_resolution,x_size,y_size,guess_ng=2.18)
				# assign guess_ng for faster simulation.
				pt_qc = get_Qc(overlap,λ,neff_ring,neff_bus,radius,gap,rw,bw,wraparound_ang)
				print('Qc =',pt_qc)
				Qc[i][j][k] += pt_qc
				progress.update(1)
	progress.close()
	print('finished simulation')
	return Qc,ex_ring,ey_ring,ex_bus,ey_bus,ring_index_arr,bus_index_arr

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# modes parameters
	parser.add_argument('-rm',type=str,default='TM',help='mode inside ring, can either be "TE" or "TM", default is "TM"')
	parser.add_argument('-rm_n',type=int,default=0,help='mode number inside ring, default of fundamental mode 0')
	parser.add_argument('-bm',type=str,default='TM',help='mode inside bus waveguide, can either be "TE" or "TM", default is "TM"')
	parser.add_argument('-bm_n',type=int,default=0,help='mode number inside bus, default of fundamental mode 0')
	parser.add_argument('-nmodes',type=int,default=10,help='number of modes for FIMMWAVE to simulate, default is 10')
	# environment parameters
	parser.add_argument('-field_resolution',type=int,default=300,help='x and y resolution of the simulation environment, default 300')
	parser.add_argument('-x_size',type=float,default=9,help='x size of the simulation environment in um, default 9')
	parser.add_argument('-y_size',type=float,default=4,help='y size of the simulation environment in um, default 9')
	# geometry parameters
	parser.add_argument('-sidewall_angle',type=float,default=10,help='sidewall angle of your waveguide in degrees')
	parser.add_argument('-gap',type=float,default=0.5,help='gap between the bus waveguide and ring in um. Calculated from the top of the waveguide')
	parser.add_argument('-radius',type=float,default=50,help='radius of the ring in um, default 50')
	parser.add_argument('-height',type=float,default=1,help='height of waveguide in um, default 1')
	parser.add_argument('-offset_x',type=float,default=1.5,help='offset of waveguide geometries (um) to make it fit into the simulation environment, positive offset causes waveguides to shift to the right, default 1.5')
	parser.add_argument('-wraparound_ang',type=float,default=20,help='angle of the wrap-around bus waveguide in degrees, default is 20')
	# vary parameters
	parser.add_argument('-rw',nargs='+',default=[2,2,1],help='ring widths (um) to scan, in the format of -rw start end points, default is 2 2 1')
	parser.add_argument('-bw',nargs='+',default=[1,1,1],help ='bus widths (um) to scan, in the format of -bw start end points, default is 1 1 1')
	parser.add_argument('-wl',nargs='+',default=[1.55,1.55,1],help='wavelengths (um) to scan, in the format of -wl start end points, default is 1.55 1.55 1')
	# material parameters
	parser.add_argument('-wg_mat',type=str,default='aln',help='choose material of waveguide: aln, sio2, al2o3, n_linbo3_z, must choose from within shapes class, you must set the material in FIMMWAVE as well')
	parser.add_argument('-sub_mat',type=str,default='al2o3',help='choose material of substrate: aln, sio2, al2o3, n_linbo3_z, must choose from within shapes class, you must set the material in FIMMWAVE as well')
	parser.add_argument('-clad_mat',type=str,default='sio2',help='choose material of cladding: aln, sio2, al2o3, n_linbo3_z, must choose from within shapes class, you must set the material in FIMMWAVE as well')
	
	args = parser.parse_args()
	ring_w = np.linspace(float(args.rw[0]),float(args.rw[1]),int(args.rw[2]))
	bus_w = np.linspace(float(args.bw[0]),float(args.bw[1]),int(args.bw[2]))
	λ = np.linspace(float(args.wl[0]),float(args.wl[1]),int(args.wl[2]))

	# ring_mode,bus_mode,ring_mode_n,bus_mode_n,Nmodes,field_resolution,x_size,y_size,
	# 			sidewall_angle,gap,radius,height,offset_x,wraparound_ang,ring_w,bus_w,pump_λ,
	# 			waveguide_material,substrate_material,cladding_material
	Qc,ex_ring,ey_ring,ex_bus,ey_bus,ring_index_arr,bus_index_arr = coupling_q(args.rm,args.bm,args.rm_n,args.bm_n,args.nmodes,
		args.field_resolution,args.x_size,args.y_size,args.sidewall_angle,args.gap,args.radius,args.height,args.offset_x,
		args.wraparound_ang,ring_w,bus_w,λ,args.wg_mat,args.sub_mat,args.clad_mat)

	# plt.figure(dpi=300)
	matplotlib.style.use('seaborn-whitegrid')
	efield_ring = {'TM':ey_ring,'TE':ex_ring}
	efield_bus = {'TM':ey_bus,'TE':ex_bus}
	fig,ax = plt.subplots(3,1,gridspec_kw={'hspace':0.25})
	figsize=(4,9)
	fig.set_size_inches(figsize)
	
	ax[0].imshow(np.abs((bus_index_arr.T==ring_index_arr.T)-ring_index_arr.T-bus_index_arr.T),origin='lower',cmap='tab20c')
	ax[0].contour(efield_ring[args.rm].T)
	ax[0].contour(efield_bus[args.bm].T)
	labels_Qc_plot = []
	labels_bw_plot = []
	for i,rw in enumerate(Qc[:]): 
		for j,bw in enumerate(Qc[0][:]):
			ax[1].plot(λ,np.log10(Qc[i][j]))
			labels_Qc_plot.append(f"rw={ring_w[i]}, bw={bus_w[j]}")
	for i in range(len(Qc)):
		for j in range(len(Qc[0][0])):
			ax[2].plot(bus_w,np.log10(Qc[i,:,j]))
			labels_bw_plot.append(f"rw={ring_w[i]}, λ={λ[j]}")

	ax[0].ticklabel_format(useOffset=False)
	ax[1].ticklabel_format(useOffset=False)
	ax[2].ticklabel_format(useOffset=False)
	ax[1].legend(labels=labels_Qc_plot,fontsize='small',loc='upper right')
	ax[2].legend(labels=labels_bw_plot,fontsize='small',loc='upper right')

	ax[0].set_title('FIMMWAVE geometry')
	ax[0].set_xlabel('x-grid')
	ax[0].set_ylabel('y-grid')
	ax[1].set_title(f'Qc plot, radius={args.radius}, wraparound={args.wraparound_ang}deg, ring to bus {args.rm}{args.rm_n} to {args.bm}{args.bm_n}',fontsize='x-small')
	ax[1].set_xlabel('Wavelength (μm)')
	ax[1].set_ylabel('$\mathrm{log_{10}}$(Qc)')
	ax[2].set_xlabel('Bus waveguide width (μm)')
	ax[2].set_ylabel('$\mathrm{log_{10}}$(Qc)')
	plt.show()