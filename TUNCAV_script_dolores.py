import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from mlpPyAPI.api import connect_to_api

### remember to change file name at end of script before each run
### remember to return fiber to (0,0,0) before each run

#final "regular design" device location (last one before ring resonators)
xf=5879.10
yf=-10.00
zf=15.30
measured=0 #(The device #, and count the 0 index)
total_devices=108 #(not counting 0 index (GDS label))
# to_measure=10

xgds1 = np.linspace(0,55*(total_devices-1-measured),total_devices-measured)
xgds2 = np.linspace(0,55*(total_devices-1-measured),total_devices-measured)+20
xgds3 = np.linspace(0,55*(total_devices-1-measured),total_devices-measured)+35

xgds=np.concatenate((xgds1,xgds2,xgds3))
xgds=-1*np.sort(xgds)
#xgds = np.linspace(0,8*(59-measured),60-measured)
ygds = np.array([0 for x in xgds])
zgds = np.array([0 for x in xgds])

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    print(kmat)
    return rotation_matrix

vec2 = [xf,yf,zf]
vec1 = [xgds[-1], 0, 0]

mat = rotation_matrix_from_vectors(vec1, vec2)
vec1_rot = mat.dot(vec1)

def instrument_check():
	laser = mlp.laser
	detec = mlp.detector

	# check and fix connection settings before testing
	if laser.is_connected() == False:
		laser.set_connection_param('gbiAddress', 1)
		laser.connect()

	if detec.is_connected() == False:
		detec.set_connection_param('gbiAddress', 1)
		detec.connect()

	if laser.is_on() == False:
		mlp.laser.turn_on()

	can_start = True



	if not mlp.fine_align.can_start():
		raise ValueError("Fine alignment cannot start. Not all required instruments are connected.")
		can_start = False
	if mlp.fine_align.get_param('algorithm') != '2d (xz)':
		raise ValueError("Fine alignment algorithm not suitable for edge coupled devices.  Ensure alignment method is '2d (xz) - spiral only")
		can_start = False
	if not mlp.laser_sweep.can_start():
		raise ValueError("Fine alignment cannot start. Not all required instruments are connected.")
		can_start = False

	return can_start


mlp=connect_to_api()

fine = mlp.fine_align
laser = mlp.laser
detec = mlp.detector

results_path = input('Please enter the path to directory where results will be saved:')

current_x, current_y,current_z = mlp.fiber_stage[0].get_position()[0], mlp.fiber_stage[0].get_position()[1], mlp.fiber_stage[0].get_position()[2]
transx = mat.dot([xgds,ygds,zgds])[0]+current_x
transy = mat.dot([xgds,ygds,zgds])[1]+current_y
transz = mat.dot([xgds,ygds,zgds])[2]+current_z

x_history = []
y_history = []
z_history = []

qq=0
for i in range(len(xgds)):
	current_x, current_y,current_z = mlp.fiber_stage[0].get_position()[0], mlp.fiber_stage[0].get_position()[1], mlp.fiber_stage[0].get_position()[2]

	dx = transx[i]-current_x
	dy = transy[i]-current_y
	dz = transz[i]-current_z

	mlp.fiber_stage[0].move_relative(dx=dx)
	print('moved '+str(dx)+' in x')

	mlp.fiber_stage[0].move_relative(dz=dz)
	print('moved '+str(dz)+' in z')

	mlp.fiber_stage[0].move_relative(dy=dy)
	print('moved '+str(dy)+' in y')


	print('testing device '+str(i+measured)+'...')
	instrument_check()

	# fine align
	fine = mlp.fine_align
	ready = fine.can_start()
	if ready:
		algo = fine.get_param('algorithm')
		if algo == '2d (xz)':
			oob=True
			count=0
			pre_x, pre_y, pre_z = mlp.fiber_stage[0].get_position()[0], mlp.fiber_stage[0].get_position()[1], mlp.fiber_stage[0].get_position()[2]

			while oob:
				fine.set_param('wavelength', 1618+2*count)

				print('Beginning Fine Align...')
				fine.start()
				print('Fine Align Complete!')
				#result = fine.get_result_data()
				#print('residuals norm = '+str(result['res_norm']))

				current_x, current_y,current_z = mlp.fiber_stage[0].get_position()[0], mlp.fiber_stage[0].get_position()[1], mlp.fiber_stage[0].get_position()[2]

				# mlp.fiber_stage[0].move_relative(dz=-.5)
				# print('moved 1')

				dx = transx[i]-current_x
				dy = transy[i]-current_y
				dz = transz[i]-current_z

				if abs(dx) >5 or abs(dy) >5 or abs(dz) >5:
					oob = True
					count+=1
					mlp.fiber_stage[0].move_relative(dx=dx)
					print('moved '+str(dx)+' in x')
					mlp.fiber_stage[0].move_relative(dz=dz)
					print('moved '+str(dz)+' in z')
					mlp.fiber_stage[0].move_relative(dy=dy)
					print('moved '+str(dy)+' in y')
				else:
					oob=False
				if count > 3:	
					mlp.fiber_stage[0].move_relative(dx=dx)
					print('moved '+str(dx)+' in x')
					mlp.fiber_stage[0].move_relative(dz=dz)
					print('moved '+str(dz)+' in z')
					mlp.fiber_stage[0].move_relative(dy=dy)
					print('moved '+str(dy)+' in y')
					print('--------------------------------------------')
					print('DEVICE'+str(i+measured)+'FAILED')
					print('--------------------------------------------')
					break

			if count < 3:
				current_x, current_y,current_z = mlp.fiber_stage[0].get_position()[0], mlp.fiber_stage[0].get_position()[1], mlp.fiber_stage[0].get_position()[2]
			else: 
				bad_x, bad_y,bad_z = mlp.fiber_stage[0].get_position()[0], mlp.fiber_stage[0].get_position()[1], mlp.fiber_stage[0].get_position()[2]
				mlp.fiber_stage[0].move_relative(dx=pre_x-bad_x)
				mlp.fiber_stage[0].move_relative(dy=pre_y-bad_y)
				mlp.fiber_stage[0].move_relative(dz=pre_z-bad_z)

			laser_ready = mlp.laser_sweep.can_start()
			if laser_ready:
				print('Starting Laser Sweep')
				mlp.laser_sweep.start()
				mlp.laser_sweep.set_save_results_path(results_path)
				if i%3 == 0:
					device_label="A2_"
				elif i%3 == 1:
					device_label="B2_"
				elif i%3 == 2:
					device_label="C2_"
				mlp.laser_sweep.save_results_to_csv(device_label+str(qq+measured)+'.csv',use_results_path=True)
			else:
				print('Laser Not Ready!')
		else:
			print('wrong fine align algo!')
	else:
		print('Fine Align not ready!')
	x_history += [mlp.fiber_stage[0].get_position()[0]]
	y_history += [mlp.fiber_stage[0].get_position()[1]]
	z_history += [mlp.fiber_stage[0].get_position()[2]]
	if i%3 ==2:
		qq+=1
	# mlp.fiber_stage[0].move_relative(dz=.5)
