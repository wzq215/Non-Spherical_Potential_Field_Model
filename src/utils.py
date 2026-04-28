import numpy as np
import pyvista as pv



def appendSpherical_np(xyz,all_postive=False):
    if np.shape(xyz) == (3,):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[0] ** 2 + xyz[1] ** 2
        ptsnew[3] = np.sqrt(xy + xyz[2] ** 2)
        # ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        ptsnew[4] = np.arctan2(xyz[2], np.sqrt(xy))  # for elevation angle defined from XY-plane up
        ptsnew[5] = np.arctan2(xyz[1], xyz[0])
        if all_postive:
            ptsnew[5][ptsnew[5]<0] += 2*np.pi
        return ptsnew
    else:
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
        # ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
        if all_postive:
            ptsnew[:,5][ptsnew[:,5]<0] += 2*np.pi

        return ptsnew


def rlonlat2line(r_Rs_vect,lon_deg_vect,lat_deg_vect,to_xyz=True):
    rlonlat = np.vstack([r_Rs_vect,
                              np.deg2rad(lon_deg_vect),
                              np.deg2rad(lat_deg_vect)])
    if to_xyz:
        xyz = np.array(rlonlat2xyz_in_Carrington(rlonlat))
    else:
        xyz = np.vstack([lon_deg_vect,lat_deg_vect,r_Rs_vect])

    line = pv.lines_from_points(np.array(xyz).T)
    return line

def rlonlat2xyz_in_Carrington(rtp_carrington, for_psi=False):
    if for_psi:
        rtp_carrington[2] = np.pi / 2 - rtp_carrington[2]

    z_carrington = rtp_carrington[0] * np.cos(np.pi / 2 - rtp_carrington[2])
    y_carrington = rtp_carrington[0] * np.sin(np.pi / 2 - rtp_carrington[2]) * np.sin(rtp_carrington[1])
    x_carrington = rtp_carrington[0] * np.sin(np.pi / 2 - rtp_carrington[2]) * np.cos(rtp_carrington[1])
    return x_carrington, y_carrington, z_carrington


def xyz2rlonlat_in_Carrington(xyz_carrington, for_psi=False):
    """
    Convert (x,y,z) to (r,t,p) in Carrington Coordination System.
        (x,y,z) follows the definition of SPP_HG in SPICE kernel.
        (r,lon,lat) is (x,y,z) converted to heliographic lon/lat, where lon \in [0,360], lat \in [-90,90] .
    :param xyz_carrington:
    :return:
    """
    r_carrington = np.linalg.norm(xyz_carrington[0:3], 2)

    lon_carrington = np.arcsin(xyz_carrington[1] / np.sqrt(xyz_carrington[0] ** 2 + xyz_carrington[1] ** 2))
    if xyz_carrington[0] < 0:
        lon_carrington = np.pi - lon_carrington
    if lon_carrington < 0:
        lon_carrington += 2 * np.pi

    lat_carrington = np.pi / 2 - np.arccos(xyz_carrington[2] / r_carrington)
    if for_psi:
        lat_carrington = np.pi / 2 - lat_carrington
    return r_carrington, np.rad2deg(lon_carrington), np.rad2deg(lat_carrington)


def dphidr(r, phi_at_r, Vsw_at_r):
    period_sunrot = 27. * (24. * 60. * 60)  # unit: s
    omega_sunrot = 2 * np.pi / period_sunrot
    result = omega_sunrot / Vsw_at_r  # unit: rad/km
    return result


def parker_spiral(r_vect_au, lat_beg_deg, lon_beg_deg, Vsw_r_vect_kmps):
    from_au_to_km = 1.49597871e8  # unit: km
    from_deg_to_rad = np.pi / 180.
    from_rs_to_km = 6.96e5
    from_au_to_rs = from_au_to_km / from_rs_to_km
    r_vect_km = r_vect_au * from_au_to_km
    num_steps = len(r_vect_km) - 1
    phi_r_vect = np.zeros(num_steps + 1)
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
    # r_footpoint_on_SourceSurface_rs = r_vect_au[-1] * from_au_to_rs
    # lon_footpoint_on_SourceSurface_deg = lon_r_vect_deg[-1]
    # lat_footpoint_on_SourceSurface_deg = lat_r_vect_deg[-1]
    return lon_r_vect_deg, lat_r_vect_deg

def spread_spherical_mesh(msh,copy_field_lst=[],copy_name_lst=[],type='surface',split_180=False,if_sort=True):
    # msh_copy = copy.copy(msh)
    xyz_rlatlon = appendSpherical_np(msh.points,all_postive=False)
    xyz_rlatlon[:,3] = xyz_rlatlon[:,3]*100.
    # msh_copy.points[:,0] = np.rad2deg(xyz_rlatlon[:,5]).ravel('F')
    # msh_copy.points[:,1] = np.rad2deg(xyz_rlatlon[:,4]).ravel('F')
    # msh_copy.points[:,2] = xyz_rlatlon[:,3].ravel('F')
    if type == 'surface':
        msh_copy = pv.PolyData(
            np.array([np.rad2deg(xyz_rlatlon[:, 5]), np.rad2deg(xyz_rlatlon[:, 4]), xyz_rlatlon[:, 3]]).T)
        msh_copy = msh_copy.delaunay_2d()
    elif type == 'line':
        if if_sort:
            sort_mask = np.argsort(xyz_rlatlon[:, 3])[::-1]
            msh_copy = pv.lines_from_points(np.array([np.rad2deg(xyz_rlatlon[sort_mask, 5]), np.rad2deg(xyz_rlatlon[sort_mask, 4]), xyz_rlatlon[sort_mask, 3]]).T)
            if split_180:
                lon_split = 180
                mask = np.abs(np.diff(np.rad2deg(xyz_rlatlon[sort_mask, 5]))) > lon_split
                msh_copy.points[np.where(mask)[0], 0] = np.nan
        else:
            msh_copy = pv.lines_from_points(np.array(
                [np.rad2deg(xyz_rlatlon[:, 5]), np.rad2deg(xyz_rlatlon[:, 4]),
                 xyz_rlatlon[:, 3]]).T)
            if split_180:
                lon_split = 180
                mask = np.abs(np.diff(np.rad2deg(xyz_rlatlon[:, 5]))) > lon_split
                msh_copy.points[np.where(mask)[0], 0] = np.nan


    for i in range(len(copy_field_lst)):
        msh_copy.point_data[copy_name_lst[i]] = msh[copy_field_lst[i]]



    return msh_copy