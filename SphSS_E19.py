import os

import pyvista
import sunpy.map
from shell_generator import shell_generator
from field_solver import fem_solver
from ss_extractor import source_surface_extractor, br_on_ss, br_on_ss_interp

PATH_2D = 'MESH/2D/'
PATH_3D = 'MESH/3D/'
PATH_RESULT = 'RESULT/'
PATH_MAGMAP = 'magnetogram/'

###### USER INPUT BEGIN ######
IB_name = 'SphR1Ref3'
OB_name = 'SphR2d5Ref3'
OOB_name = 'SphR10Ref3'

magmap_name_list = [ 'mrzqs240331t2104c2282_050', 'mrzqs240331t0304c2282_060',
                    'mrzqs240330t0914c2282_070', 'mrzqs240329t1514c2282_080',
                     'mrzqs240328t2004c2282_090','mrzqs240328t0204c2282_100']

for i in range(6):
    magmap_name = magmap_name_list[i]
    magmap_tag = magmap_name[-9:]
    PATH_RESULT = 'RESULT/' + 'CR2282_E19/SphSS_Rss1d9_Ref3/'+magmap_tag[-3:]+'/'

    MS_radius = 1.9
    # SS_Br = 25000e-5
    ss_tag = ''
    ###### USER INPUT END ######

    os.makedirs(PATH_RESULT,exist_ok=True)

    # %%
    shell0_path, shell0_name, readable = shell_generator(OB_name, IB_name, refine_level=0,
                                                         path_2D=PATH_2D, path_3D=PATH_3D)

    # %%
    result0_path, result0_name, result0 = fem_solver(shell0_path, shell0_name,
                                                     magmap_method='interp', abs_field=False,
                                                     magmap_pathfilename=PATH_MAGMAP+magmap_name+'.fits.gz',
                                                     magmap_tag=magmap_tag,
                                                     result_path=PATH_RESULT, )
    # quit()
    # %%
    ss0_path = PATH_2D
    ss0_name = OB_name
    shell1_path = shell0_path
    shell1_name = shell0_name
    result1_path = result0_path
    result1_name = result0_name
    result1=result0

    ss_xyz_rlatlon, br_ss, clm_ss = br_on_ss(result1_name, ss0_name, l_max=30, path_result=result0_path)


    shell2_path, shell2_name, readable = shell_generator(OOB_name, ss0_name, refine_level=0,
                                                         path_2D=PATH_2D, path_3D=PATH_3D)

    # %%
    result2_path, result2_name, result2 = fem_solver(shell2_path, shell2_name,
                                                     magmap_method='array',
                                                     # const_br=abs(br_ss),
                                                     abs_field=True,
                                                     magmap_tag='array', result_path=PATH_RESULT,
                                                     clm=clm_ss, l_max=30)

