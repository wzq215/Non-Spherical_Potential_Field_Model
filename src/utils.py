"""
utils.py

Utility functions for coordinate transformations, magnetic field-line
construction, and geometric post-processing used in the Non-Spherical
Potential Field (NSPF) model workflow.

This module provides:
- Cartesian and spherical (r, lon, lat) coordinate conversions in the
  Carrington heliographic system.
- Helper functions for constructing PyVista line objects from spherical
  coordinates.
- Parker spiral trajectory computation for solar wind connectivity.


These functions are used throughout the NSPF solver, post-processing,
and visualization routines. This module is intended to be imported
as a library and not executed as a standalone program.

Associated publication:
Wu et al. (2026), ApJ, submitted.

Author: Ziqi Wu
"""

import numpy as np
import pyvista as pv

def appendSpherical_np(xyz,all_postive=False):
    """
    Append spherical coordinates to Cartesian coordinates.

    Given one or more Cartesian points (x, y, z), this function computes
    the corresponding spherical coordinates (r, elevation, longitude)
    and appends them to the original array.

    Parameters
    ----------
    xyz : array-like
        Cartesian coordinates of shape (3,) or (N, 3).
    all_postive : bool, optional
        If True, enforce longitude values in the range [0, 2π).

    Returns
    -------
    ptsnew : ndarray
        Array containing Cartesian and spherical coordinates:
        [x, y, z, r, elevation, longitude] for each input point.
    """

    # --- Single point input ---
    if np.shape(xyz) == (3,):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[0] ** 2 + xyz[1] ** 2
        ptsnew[3] = np.sqrt(xy + xyz[2] ** 2) # radius r
        ptsnew[4] = np.arctan2(xyz[2], np.sqrt(xy))  # elevation angle defined from XY-plane up
        ptsnew[5] = np.arctan2(xyz[1], xyz[0]) # longitude
        if all_postive:
            ptsnew[5][ptsnew[5]<0] += 2*np.pi
        return ptsnew
    # --- Multiple points input ---
    else:
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
        ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
        if all_postive:
            ptsnew[:,5][ptsnew[:,5]<0] += 2*np.pi

        return ptsnew


def rlonlat2line(r_Rs_vect,lon_deg_vect,lat_deg_vect,to_xyz=True):
    """
        Construct a PyVista line object from spherical coordinates.

        Parameters
        ----------
        r_Rs_vect : array-like
            Radial distances (in solar radii).
        lon_deg_vect : array-like
            Carrington longitudes in degrees.
        lat_deg_vect : array-like
            Carrington latitudes in degrees.
        to_xyz : bool, optional
            If True, convert to Cartesian coordinates before constructing the line.

        Returns
        -------
        line : pyvista.PolyData
            PolyData line representing the input trajectory.
        """

    rlonlat = np.vstack([r_Rs_vect,
                              np.deg2rad(lon_deg_vect),
                              np.deg2rad(lat_deg_vect)])
    if to_xyz:
        xyz = np.array(rlonlat2xyz_in_Carrington(rlonlat))
    else:
        xyz = np.vstack([lon_deg_vect,lat_deg_vect,r_Rs_vect])

    line = pv.lines_from_points(np.array(xyz).T)
    return line

def rlonlat2xyz_in_Carrington(rtp_carrington, is_colat=False):
    """
        Convert spherical coordinates to Cartesian coordinates in the
        Carrington heliographic coordinate system.

        Parameters
        ----------
        rtp_carrington : array-like
            Array containing (r, lon, lat) in radians.
        is_colat : bool, optional
            If True, the input lat is treated as a colatitude.

        Returns
        -------
        x, y, z : ndarray
            Cartesian Carrington coordinates.
        """

    if is_colat:
        rtp_carrington[2] = np.pi / 2 - rtp_carrington[2]

    z_carrington = rtp_carrington[0] * np.cos(np.pi / 2 - rtp_carrington[2])
    y_carrington = rtp_carrington[0] * np.sin(np.pi / 2 - rtp_carrington[2]) * np.sin(rtp_carrington[1])
    x_carrington = rtp_carrington[0] * np.sin(np.pi / 2 - rtp_carrington[2]) * np.cos(rtp_carrington[1])
    return x_carrington, y_carrington, z_carrington


def xyz2rlonlat_in_Carrington(xyz_carrington, use_colat=False):
    """
    Convert Cartesian Carrington coordinates to spherical coordinates.

    Parameters
    ----------
    xyz_carrington : array-like
        Cartesian coordinates (x, y, z).
    use_colat : bool, optional
        If True, return colatitude instead of latitude.

    Returns
    -------
    r : float
        Radial distance.
    lon : float
        Carrington longitude in degrees [0, 360).
    lat : float
        Carrington latitude in degrees.
    """

    r_carrington = np.linalg.norm(xyz_carrington[0:3], 2)

    lon_carrington = np.arcsin(xyz_carrington[1] / np.sqrt(xyz_carrington[0] ** 2 + xyz_carrington[1] ** 2))
    if xyz_carrington[0] < 0:
        lon_carrington = np.pi - lon_carrington
    if lon_carrington < 0:
        lon_carrington += 2 * np.pi

    lat_carrington = np.pi / 2 - np.arccos(xyz_carrington[2] / r_carrington)
    if use_colat:
        lat_carrington = np.pi / 2 - lat_carrington
    return r_carrington, np.rad2deg(lon_carrington), np.rad2deg(lat_carrington)


def dphidr(r, phi_at_r, Vsw_at_r):
    """
    Compute the radial derivative of the Parker spiral azimuthal angle.

    Parameters
    ----------
    r : float
        Radial distance.
    phi_at_r : float
        Azimuthal angle at radius r (radians).
    Vsw_at_r : float
        Solar wind speed at radius r (km/s).

    Returns
    -------
    dphi_dr : float
        Azimuthal derivative with respect to radius.
    """

    period_sunrot = 27. * (24. * 60. * 60)  # unit: s
    omega_sunrot = 2 * np.pi / period_sunrot
    result = omega_sunrot / Vsw_at_r  # unit: rad/km
    return result


def parker_spiral(r_vect_au, lat_beg_deg, lon_beg_deg, Vsw_r_vect_kmps):
    """
    Compute the radial derivative of the Parker spiral azimuthal angle.

    Parameters
    ----------
    r : float
        Radial distance.
    phi_at_r : float
        Azimuthal angle at radius r (radians).
    Vsw_at_r : float
        Solar wind speed at radius r (km/s).

    Returns
    -------
    dphi_dr : float
        Azimuthal derivative with respect to radius.
    """

    from_au_to_km = 1.49597871e8  # unit: km
    from_deg_to_rad = np.pi / 180.
    from_rs_to_km = 6.96e5
    from_au_to_rs = from_au_to_km / from_rs_to_km
    r_vect_km = r_vect_au * from_au_to_km
    num_steps = len(r_vect_km) - 1
    phi_r_vect = np.zeros(num_steps + 1)
    #  RK4 integration of Parker spiral equation
    for i_step in range(0, num_steps):
        if i_step == 0:
            phi_at_r_current = lon_beg_deg * from_deg_to_rad  # unit: rad
            phi_r_vect[0] = phi_at_r_current
        else:
            phi_at_r_current = phi_at_r_next
        r_current = r_vect_km[i_step]
        r_next = r_vect_km[i_step + 1]
        r_mid = (r_current + r_next) / 2
        dr = r_current - r_next
        Vsw_at_r_current = Vsw_r_vect_kmps[i_step - 1]
        Vsw_at_r_next = Vsw_r_vect_kmps[i_step]
        Vsw_at_r_mid = (Vsw_at_r_current + Vsw_at_r_next) / 2
        k1 = dr * dphidr(r_current, phi_at_r_current, Vsw_at_r_current)
        k2 = dr * dphidr(r_current + 0.5 * dr, phi_at_r_current + 0.5 * k1, Vsw_at_r_mid)
        k3 = dr * dphidr(r_current + 0.5 * dr, phi_at_r_current + 0.5 * k2, Vsw_at_r_mid)
        k4 = dr * dphidr(r_current + 1.0 * dr, phi_at_r_current + 1.0 * k3, Vsw_at_r_next)
        phi_at_r_next = phi_at_r_current + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi_r_vect[i_step + 1] = phi_at_r_next
    lon_r_vect_deg = phi_r_vect / from_deg_to_rad  # from [rad] to [degree]
    lat_r_vect_deg = np.zeros(num_steps + 1) + lat_beg_deg  # unit: [degree]
    return lon_r_vect_deg, lat_r_vect_deg
