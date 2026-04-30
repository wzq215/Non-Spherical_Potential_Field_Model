"""
sourcesurface_extractor.py

Utilities for extracting, smoothing, and analyzing the magnetic source
surface (SS) from Non-Spherical Potential Field (NSPF) simulation results.

This module provides functions to:
- Extract a source surface as an isosurface of magnetic field magnitude.
- Smooth the extracted SS using spherical harmonic expansion.
- Reconstruct SS geometry on a reference spherical mesh.
- Evaluate magnetic field strength (Br or |B|) on the SS.
- Generate gridded magnetic field maps on the SS for further analysis.

These routines are primarily intended for post-processing and analysis
of NSPF results, and for constructing SS surfaces used in subsequent
field-line tracing and visualization.

This module is designed to be imported and used by higher-level analysis
scripts and is not intended to be executed as a standalone program.

Associated publication:
Wu et al. (2026), ApJ, submitted.

Author: Ziqi Wu
"""

import numpy as np
import pyshtools as shtools
import pyvista

from src.utils import appendSpherical_np


def source_surface_extractor(result_path, result_name,
                             ss_Btot=-1, Rss=2.5, ss_tag='',
                             smooth_lmax=10,
                             path_2D='MESH/2D/', sph_ini_name=''):
    """
    Extract and smooth the magnetic source surface from an NSPF solution.

    The source surface (SS) is defined as an isosurface of magnetic field
    magnitude |B| near a specified source-surface radius. The extracted
    surface is smoothed using spherical harmonic expansion and exported
    as an STL mesh for subsequent use.

    Parameters
    ----------
    result_path : str
        Directory containing the NSPF VTK result file.
    result_name : str
        File name (without extension) of the NSPF result.
    ss_Btot : float, optional
        Target magnetic field magnitude defining the source surface.
        If set to -1, the value is automatically inferred near Rss.
    Rss : float, optional
        Approximate source surface radius (in solar radii).

    ss_tag : str, optional
        Tag appended to the exported source-surface file name.
    smooth_lmax : int, optional
        Maximum spherical harmonic degree used for smoothing.
    path_2D : str, optional
        Output directory for 2D surface meshes.
    sph_ini_name : str, optional
        File name of the initial spherical reference surface (STL).

    Returns
    -------
    path_2D : str
        Directory where the source surface mesh is saved.
    exportname : str
        Base name of the exported source surface.
    ss_smooth : pyvista.PolyData
        Smoothed source surface mesh.
    """

    # --- Load NSPF magnetic field result and determine SS isovalue ---
    mesh_b = pyvista.read(result_path + result_name + '.vtk')
    mesh_b.set_active_scalars('Btot')
    if ss_Btot == -1:
        # Automatically infer SS |B| near the target source-surface radius
        ss_Btot = np.max(
            mesh_b['Btot'][
                (mesh_b.points[:, 0] ** 2 + mesh_b.points[:, 1] ** 2 + mesh_b.points[:, 2] ** 2) >= (Rss - 0.1) ** 2])
    # --- Extract source surface as an isosurface of |B| ---
    contours = mesh_b.contour(scalars='Btot', isosurfaces=1, rng=[ss_Btot, ss_Btot])

    ss = contours.connectivity(largest=True)
    ss.clear_field_data()
    exportname = result_name + '_SS' + ss_tag
    # --- Smooth the extracted source surface using spherical harmonics ---
    ss_smooth, ss_clm = smooth_sourcesurface(ss, sph_ini_name,
                                             l_max=smooth_lmax, path_2D=path_2D)

    # --- Save smoothed SS geometry and spherical harmonic coefficients ---
    ss_smooth.save(path_2D + exportname + '.stl')
    print('Saving smoothed surface to: ' + path_2D + exportname + '.stl')

    np.save('SPHs/SS/' + exportname, ss_clm)
    print('Saving clm to: ', 'SPHs/SS/' + exportname + '.npy')

    return path_2D, exportname, ss_smooth


def smooth_sourcesurface(ss_raw, sph_ini_path, l_max=10,
                         path_2D='MESH/2D/', ):
    """
    Smooth a raw source surface using spherical harmonic expansion.

    The raw SS mesh is expanded in spherical harmonics, filtered via a
    maximum degree cutoff, and reconstructed onto a reference spherical
    surface to obtain a smooth and well-behaved geometry.

    Parameters
    ----------
    ss_raw : pyvista.PolyData
        Raw source surface mesh extracted from NSPF results.
    sph_ini_path : str
        File name of the reference spherical mesh (STL).
    l_max : int, optional
        Maximum spherical harmonic degree used for smoothing.
    path_2D : str, optional
        Directory containing the reference spherical mesh.

    Returns
    -------
    surface_new : pyvista.PolyData
        Smoothed and reconstructed source surface.
    clm : ndarray
        Spherical harmonic coefficients of the SS radius.
    """

    # --- Convert raw SS points to spherical coordinates ---
    xyz_rlatlon_raw = appendSpherical_np(ss_raw.points)
    # --- Least-squares spherical harmonic expansion of SS radius
    clm, chi2 = shtools.expand.SHExpandLSQ(xyz_rlatlon_raw[:, 3], np.rad2deg(xyz_rlatlon_raw[:, 4]),
                                           np.rad2deg(xyz_rlatlon_raw[:, 5]), lmax=l_max)
    print('Minimum r_ss in raw SS: ', np.nanmin(xyz_rlatlon_raw[:,3]))
    # --- Smooth original SS mesh using reconstructed harmonic field ---
    value = shtools.expand.MakeGridPoint(clm, np.rad2deg(xyz_rlatlon_raw[:, 4]), np.rad2deg(xyz_rlatlon_raw[:, 5]))
    surface_smooth = ss_raw
    surface_smooth.points = (surface_smooth.points.T * value / xyz_rlatlon_raw[:, 3]).T
    # --- Reconstruct SS on a reference spherical mesh ---
    surface_new = pyvista.read(path_2D + sph_ini_path + '.stl')
    surface_new_xyz_rlatlon = appendSpherical_np(surface_new.points)
    print('Minimum r_ss in new SS: ', np.nanmin(surface_new_xyz_rlatlon[:, 3]))
    surface_new_value = shtools.expand.MakeGridPoint(clm, np.rad2deg(surface_new_xyz_rlatlon[:, 4]),
                                                     np.rad2deg(surface_new_xyz_rlatlon[:, 5]))
    surface_new.points = (surface_new.points.T * surface_new_value / surface_new_xyz_rlatlon[:, 3]).T
    # --- Safety check: correct unphysical points with too small radius
    ss_rs = np.linalg.norm(surface_new.points, axis=1)
    print('Minimum r_ss in reconstructed SS: ', np.nanmin(ss_rs))
    print(ss_rs)
    bad_points_lst = np.argwhere(ss_rs<1.).reshape(-1)
    print(bad_points_lst)
    for bad_point_idx in bad_points_lst:
        bad_point = surface_new.points[bad_point_idx]
        print('Bad Point Found: ', bad_point, '. Radius: ', ss_rs[bad_point_idx])
        surface_new.points[bad_point_idx,:] = bad_point*1.01/ss_rs[bad_point_idx]
        print('Modify to: ', surface_new.points[bad_point_idx])
    return surface_new, clm


def br_on_ss(filename_result, filename_ss,
             l_max=30,
             path_2D='MESH/2D/', path_result='RESULT/', save_vtk=True):
    """
    Evaluate magnetic field strength on a source surface and fit spherical
    harmonic coefficients.

    Parameters
    ----------
    filename_result : str
        NSPF result file name (without extension).
    filename_ss : str
        Source surface file name (without extension).
    l_max : int, optional
        Maximum spherical harmonic degree for the fit.
    path_2D : str, optional
        Directory containing the SS mesh.
    path_result : str, optional
        Directory containing the NSPF result.
    save_vtk : bool, optional
        If True, save sampled SS mesh as a VTK file.

    Returns
    -------
    ss_xyz_rlatlon : ndarray
        Spherical coordinates of SS points.
    value : ndarray
        Reconstructed magnetic field on the SS.
    clm : ndarray
        Spherical harmonic coefficients of the SS field.
    """

    # --- Sample magnetic field solution onto the source surface ---
    mesh_b = pyvista.read(path_result + filename_result + '.vtk')
    ss = pyvista.read(path_2D + filename_ss + '.stl')
    mesh_ss = ss.sample(mesh_b)

    # --- Spherical harmonic expansion of |B| on the source surface
    ss_xyz_rlatlon = appendSpherical_np(mesh_ss.points)
    clm, chi2 = shtools.expand.SHExpandLSQ(mesh_ss['Btot'], np.rad2deg(ss_xyz_rlatlon[:, 4]),
                                           np.rad2deg(ss_xyz_rlatlon[:, 5]), lmax=l_max)
    value = shtools.expand.MakeGridPoint(clm, np.rad2deg(ss_xyz_rlatlon[:, 4]), np.rad2deg(ss_xyz_rlatlon[:, 5]))
    if save_vtk:
        mesh_ss.save(path_2D + filename_ss + '.vtk')

    return ss_xyz_rlatlon, value, clm



def br_on_ss_interp(filename_result, filename_ss,
                    path_result='RESULT/',):
    """
    Interpolate magnetic field values onto a dense source surface grid
    using precomputed spherical harmonic coefficients.

    Parameters
    ----------
    filename_result : str
        NSPF result file name (without extension).
    filename_ss : str
        Source surface identifier.
    path_result : str, optional
        Directory containing NSPF results.
    save_clm : bool, optional
        Reserved for future use.

    Returns
    -------
    ss_lon : ndarray
        Longitudes of the SS grid (degrees).
    ss_lat : ndarray
        Latitudes of the SS grid (degrees).
    ss_Btot : ndarray
        Magnetic field magnitude mapped onto the SS grid.
    """

    # --- Load NSPF result and SS spherical harmonic coefficients ---
    mesh_b = pyvista.read(path_result + filename_result + '.vtk')
    ss_clm = np.load('SPHs/SS/' + filename_ss + '.npy')

    # --- Reconstruct SS geometry on a dense spherical grid ---
    ss_new = pyvista.Sphere(theta_resolution=360, phi_resolution=180)
    ss_new_xyz_rlatlon = appendSpherical_np(ss_new.points)
    ss_new_height = shtools.expand.MakeGridPoint(ss_clm,
                                                 np.rad2deg(ss_new_xyz_rlatlon[:, 4]),
                                                 np.rad2deg(ss_new_xyz_rlatlon[:, 5]))
    ss_new.points = (ss_new.points.T * ss_new_height / ss_new_xyz_rlatlon[:, 3]).T
    mesh_ss_new = ss_new.sample(mesh_b)
    # --- Construct regular longitude-latitude grids of SS magnetic field ---
    ss_lon = np.arange(180., -180., -1.)
    ss_lon[ss_lon < 0] = ss_lon[ss_lon < 0] + 360
    ss_lat = np.arange(-90., 90., 1.)
    ss_LON, ss_LAT = np.meshgrid(ss_lon, ss_lat)
    ss_Btot = np.zeros_like(ss_LON)
    ss_Btot[1:-1, :] = mesh_ss_new['Btot'][2:].reshape(360, 180 - 2).T
    ss_Btot[0, :] = np.linspace(mesh_ss_new['Btot'][0], mesh_ss_new['Btot'][0], 360)
    ss_Btot[-1, :] = np.linspace(mesh_ss_new['Btot'][1], mesh_ss_new['Btot'][1], 360)

    return ss_lon, ss_lat, ss_Btot
