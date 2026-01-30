import numpy as np
import pyvista

import ufl
import pandas as pd
import math
import sympy as sp
from ufl import sqrt, sin, cos
import sunpy.io._fits as sfits


def dipole_field(x, b0=100., **kwargs):
    print('Generating Dipole field with B0=' + str(b0) + '(nT)')
    return -b0 / (4 * np.pi * 1e-7) * x[2] / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)

def magmap_interp_notwork(x, magmap_file, magmap_path='/Users/ephe/PFSS_Data/',**kwargs):
    # (x, y, z) of the 4 corner points which are given
    magmap_data, magmap_header = sfits.read(magmap_path + magmap_file + '.fits')[0]
    magmap_lon = np.linspace(0.,360.,360)
    magmap_lat = np.linspace(-90.,90.,180)
    lonlon,latlat = np.meshgrid(magmap_lon,magmap_lat)
    print('Generating magmap by interp')


    # looking for a function z(x, y) = ax + bxy + cy + d
    a, b, c, d = sp.var("a b c d")  # unknown coeffs
    rows, cols = lonlon.shape
    eqs = []
    # creating an equation for each of the point
    for i_row in range(rows):
        for i_col in range(cols):
            eqs.append(
                a * lonlon[i_row, i_col] + b * lonlon[i_row, i_col] * latlat[i_row, i_col] + c * latlat[i_row, i_col] + d - magmap_data[i_row, i_col])
    sols = sp.solve(eqs, [a, b, c, d])
    # print(sols)
    # {a: -1.00000000000000, b: -1.00000000000000, c: 1.00000000000000, d: 1.00000000000000}
    x_ = sp.Symbol("x[0]")
    y_ = sp.Symbol("x[1]")
    # resulting symbolic function
    f_symbolic = sp.Lambda((x_, y_), sols[a] * x_ + sols[b] * x_ * y_ + sols[c] * y_ + sols[d])
    return f_symbolic

def magmap_interp(x, magmap_file, magmap_path='/Users/ephe/PFSS_Data/',**kwargs):
    # (x, y, z) of the 4 corner points which are given
    magmap_data, magmap_header = sfits.read(magmap_path + magmap_file + '.fits')[0]
    magmap_lon = np.linspace(0.,360.,360)
    magmap_lat = np.linspace(-90.,90.,180)
    lonlon,latlat = np.meshgrid(magmap_lon,magmap_lat)
    print('Generating magmap by interp')
    from scipy import interpolate
    f_interp = interpolate.interp2d(magmap_lon,magmap_lat,magmap_data,kind='linear')

    x_ = sp.Symbol("x[0]")
    y_ = sp.Symbol("x[1]")
    # resulting symbolic function
    f_symbolic = sp.Lambda((x_, y_), f_interp(x_,y_))
    return f_symbolic

def magmap_from_file(x, sph_file, l_max=60, **kwargs):
    print('Generating magmap from sph file: ' + sph_file + ' . Lmax=' + str(l_max))
    sphs_df = pd.read_csv(sph_file, header=None, sep='\s+', names=['l', 'm', 'glm', 'hlm'])
    glm_array = np.zeros((60 + 1, 60 + 1))
    hlm_array = np.zeros((60 + 1, 60 + 1))
    for i in range(sphs_df.shape[0]):
        glm_array[int(sphs_df['l'][i]), int(sphs_df['m'][i])] = sphs_df['glm'][i]
        hlm_array[int(sphs_df['l'][i]), int(sphs_df['m'][i])] = sphs_df['hlm'][i]

    cos_colat = x[2] / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)
    lon = ufl.acos(x[0] / ufl.sqrt(x[0] ** 2 + x[1] ** 2)) * ufl.sign(x[1]) + ufl.pi - ufl.sign(x[1]) * ufl.pi

    COSCOLAT, LON = sp.symbols('cos_colat,lon')

    Br = sp.symbols('0.')
    for i_l in range(l_max + 1):
        for i_m in range(i_l + 1):
            # print('l: ', i_l, '; m:', i_m)
            Plm = sp.N(sp.assoc_legendre(i_l, i_m, COSCOLAT) \
                       * (2 * math.factorial(i_l - i_m)
                          / math.factorial(i_l + i_m)) ** (1 / 2))

            Br = sp.N(Br + Plm * (glm_array[i_l, i_m] * sp.cos(i_m * LON)
                                  + hlm_array[i_l, i_m] * sp.sin(i_m * LON)))
    magmap = eval(str(Br))
    return magmap

def magmap_from_array(x, clm=[], l_max=30, **kwargs):
    print('Generating magmap from input sph_array. Lmax=' + str(l_max))
    glm_array = clm[0, :, :]
    hlm_array = clm[1, :, :]

    cos_colat = x[2] / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** (1 / 2)
    lon = ufl.acos(x[0] / ufl.sqrt(x[0] ** 2 + x[1] ** 2)) * ufl.sign(x[1]) + ufl.pi - ufl.sign(x[1]) * ufl.pi

    COSCOLAT, LON = sp.symbols('cos_colat,lon')

    Br = sp.symbols('0.')
    for i_l in range(l_max + 1):
        for i_m in range(i_l + 1):
            print('l: ', i_l, '; m:', i_m)
            Plm = sp.N(sp.assoc_legendre(i_l, i_m, COSCOLAT) \
                       * (2 * math.factorial(i_l - i_m)
                          / math.factorial(i_l + i_m)) ** (1 / 2))

            Br = sp.N(Br + Plm * (glm_array[i_l, i_m] * sp.cos(i_m * LON)
                                  + hlm_array[i_l, i_m] * sp.sin(i_m * LON)))
    magmap = eval(str(Br))
    return magmap


def magmap_generator(x,
                     method='file', abs_field=False,br_ss=100.,
                     **kwargs, ):
    """

    :param x:
    :param method:
    :param abs_field:
    :param b0: (optional, for method=dipole)
    :param sph_file: (optional, for method=sph_file)
    :param l_max: (optional, for method=sph_file or sph_array)
    :param clm: (optional, for method=sph_array)
    :return:
    """

    if method == 'constant':
        magmap = x[0] * 0. + br_ss
        print('Generating constant magmap ('+str(br_ss)+' nT).')
    elif method == 'dipole':
        magmap = dipole_field(x, **kwargs)
    elif method == 'file':
        magmap = magmap_from_file(x, **kwargs)
    elif method == 'array':
        magmap = magmap_from_array(x, **kwargs)
    elif method == 'interp':
        magmap = magmap_interp(x,**kwargs)
    if abs_field:
        print('ABS IT.')
        magmap = abs(magmap)
    return magmap

