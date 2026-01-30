import numpy as np
import pyshtools as shtools
import pyvista

from utils import appendSpherical_np


def source_surface_extractor(result_path, result_name,
                             ss_Btot=-1, Rss=2.5, ss_tag='',
                             smooth_lmax=10,
                             path_2D='MESH/2D/', sph_ini_name=''):
    mesh_b = pyvista.read(result_path + result_name + '.vtk')
    mesh_b.set_active_scalars('Btot')
    if ss_Btot == -1:
        ss_Btot = np.max(
            mesh_b['Btot'][
                (mesh_b.points[:, 0] ** 2 + mesh_b.points[:, 1] ** 2 + mesh_b.points[:, 2] ** 2) >= (Rss - 0.1) ** 2])
    contours = mesh_b.contour(scalars='Btot', isosurfaces=1, rng=[ss_Btot, ss_Btot])

    ss = contours.connectivity(largest=True)
    ss.clear_field_data()
    exportname = result_name + '_SS' + ss_tag
    ss_smooth, ss_clm = smooth_sourcesurface(ss, sph_ini_name,
                                             l_max=smooth_lmax, path_2D=path_2D)

    ss_smooth.save(path_2D + exportname + '.stl')
    print('Saving smoothed surface to: ' + path_2D + exportname + '.stl')

    np.save('SPHs/SS/' + exportname, ss_clm)
    print('Saving clm to: ', 'SPHs/SS/' + exportname + '.npy')

    return path_2D, exportname, ss_smooth


def smooth_sourcesurface(ss_raw, sph_ini_path, l_max=10,
                         path_2D='MESH/2D/', ):
    xyz_rlatlon_raw = appendSpherical_np(ss_raw.points)

    clm, chi2 = shtools.expand.SHExpandLSQ(xyz_rlatlon_raw[:, 3], np.rad2deg(xyz_rlatlon_raw[:, 4]),
                                           np.rad2deg(xyz_rlatlon_raw[:, 5]), lmax=l_max)
    print('Minimum r_ss in raw SS: ', np.nanmin(xyz_rlatlon_raw[:,3]))
    value = shtools.expand.MakeGridPoint(clm, np.rad2deg(xyz_rlatlon_raw[:, 4]), np.rad2deg(xyz_rlatlon_raw[:, 5]))
    surface_smooth = ss_raw
    surface_smooth.points = (surface_smooth.points.T * value / xyz_rlatlon_raw[:, 3]).T
    # %%
    surface_new = pyvista.read(path_2D + sph_ini_path + '.stl')
    surface_new_xyz_rlatlon = appendSpherical_np(surface_new.points)
    print('Minimum r_ss in new SS: ', np.nanmin(surface_new_xyz_rlatlon[:, 3]))
    surface_new_value = shtools.expand.MakeGridPoint(clm, np.rad2deg(surface_new_xyz_rlatlon[:, 4]),
                                                     np.rad2deg(surface_new_xyz_rlatlon[:, 5]))
    surface_new.points = (surface_new.points.T * surface_new_value / surface_new_xyz_rlatlon[:, 3]).T
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
    mesh_b = pyvista.read(path_result + filename_result + '.vtk')

    ss = pyvista.read(path_2D + filename_ss + '.stl')

    mesh_ss = ss.sample(mesh_b)


    ss_xyz_rlatlon = appendSpherical_np(mesh_ss.points)
    clm, chi2 = shtools.expand.SHExpandLSQ(mesh_ss['Btot'], np.rad2deg(ss_xyz_rlatlon[:, 4]),
                                           np.rad2deg(ss_xyz_rlatlon[:, 5]), lmax=l_max)
    value = shtools.expand.MakeGridPoint(clm, np.rad2deg(ss_xyz_rlatlon[:, 4]), np.rad2deg(ss_xyz_rlatlon[:, 5]))
    # if save_clm:
    #     np.save('SPHs/SS_BR/' + filename_result + '_BRSS', clm)
    #     print('Saving clm to: ', 'SPHs/SS_BR/' + filename_result + '_BRSS.npy')
    if save_vtk:
        mesh_ss.save(path_2D + filename_ss + '.vtk')

    return ss_xyz_rlatlon, value, clm


def br_on_is(filename_result, filename_ss,
             l_max=30, fraction=0.9,
             path_2D='MESH/2D/', path_result='RESULT/', save_vtk=True):
    mesh_b = pyvista.read(path_result + filename_result + '.vtk')

    ss = pyvista.read(path_2D + filename_ss + '.stl')

    # if fraction == 1.:
    #     filename_is = filename_ss
    # else:
    ss.points = ss.points * fraction
    filename_is = filename_ss + '_IS*' + str(fraction)

    mesh_is = ss.sample(mesh_b)

    ss_xyz_rlatlon = appendSpherical_np(mesh_is.points)
    mesh_is = mesh_is.compute_normals(cell_normals=False)
    mesh_is['Bn'] = np.array(
        [np.dot(mesh_is['Bxyz'][i, :], mesh_is['Normals'][i, :]) / np.linalg.norm(mesh_is['Normals'][i, :])
         for i in range(mesh_is.n_points)])

    clm, chi2 = shtools.expand.SHExpandLSQ(mesh_is['Bn'], np.rad2deg(ss_xyz_rlatlon[:, 4]),
                                           np.rad2deg(ss_xyz_rlatlon[:, 5]), lmax=l_max)
    value = shtools.expand.MakeGridPoint(clm, np.rad2deg(ss_xyz_rlatlon[:, 4]), np.rad2deg(ss_xyz_rlatlon[:, 5]))

    mesh_is.save(path_2D + filename_is + '.stl')
    if save_vtk:
        mesh_is.save(path_2D + filename_is + '.vtk')

    return filename_is, ss_xyz_rlatlon, value, clm


def br_on_ss_interp(filename_result, filename_ss,
                    # l_max=30,
                    # path_2D='MESH/2D/',
                    path_result='RESULT/',
                    save_clm=True):
    mesh_b = pyvista.read(path_result + filename_result + '.vtk')

    # ss = pyvista.read(path_2D + filename_ss + '.stl')
    ss_clm = np.load('SPHs/SS/' + filename_ss + '.npy')

    ss_new = pyvista.Sphere(theta_resolution=360, phi_resolution=180)
    ss_new_xyz_rlatlon = appendSpherical_np(ss_new.points)
    ss_new_height = shtools.expand.MakeGridPoint(ss_clm,
                                                 np.rad2deg(ss_new_xyz_rlatlon[:, 4]),
                                                 np.rad2deg(ss_new_xyz_rlatlon[:, 5]))
    ss_new.points = (ss_new.points.T * ss_new_height / ss_new_xyz_rlatlon[:, 3]).T
    mesh_ss_new = ss_new.sample(mesh_b)

    ss_lon = np.arange(180., -180., -1.)
    ss_lon[ss_lon < 0] = ss_lon[ss_lon < 0] + 360
    ss_lat = np.arange(-90., 90., 1.)
    ss_LON, ss_LAT = np.meshgrid(ss_lon, ss_lat)
    ss_Btot = np.zeros_like(ss_LON)
    ss_Btot[1:-1, :] = mesh_ss_new['Btot'][2:].reshape(360, 180 - 2).T
    ss_Btot[0, :] = np.linspace(mesh_ss_new['Btot'][0], mesh_ss_new['Btot'][0], 360)
    ss_Btot[-1, :] = np.linspace(mesh_ss_new['Btot'][1], mesh_ss_new['Btot'][1], 360)

    # ss_xyz_rlatlon = appendSpherical_np(mesh_ss_new.points)
    # clm, chi2 = shtools.expand.SHExpandLSQ(mesh_ss_new['Btot'], np.rad2deg(ss_xyz_rlatlon[:, 4]),
    #                                        np.rad2deg(ss_xyz_rlatlon[:, 5]), lmax=l_max)
    # value = shtools.expand.MakeGridPoint(clm, np.rad2deg(ss_xyz_rlatlon[:, 4]), np.rad2deg(ss_xyz_rlatlon[:, 5]))
    # if save_clm:
    #     np.save('SPHs/SS_BR/' + filename_result + '_BRSS', clm)
    #     print('Saving clm to: ', 'SPHs/SS_BR/' + filename_result + '_BRSS.npy')
    return ss_lon, ss_lat, ss_Btot


def magflux_on_ss(filename_result, filename_ss,
                  l_max=30,
                  path_2D='MESH/2D/', path_result='RESULT/'):
    mesh_b = pyvista.read(path_result + filename_result + '.vtk')

    ss = pyvista.read(path_2D + filename_ss + '.stl')
    ss_xyz_rlatlon = appendSpherical_np(ss.points)

    sph = pyvista.Sphere(1, theta_resolution=180, phi_resolution=180)
    sph_xyz_rlatlon = appendSpherical_np(sph.points)

    ss_clm, ss_chi2 = shtools.expand.SHExpandLSQ(ss_xyz_rlatlon[:, 3], np.rad2deg(ss_xyz_rlatlon[:, 4]),
                                                 np.rad2deg(ss_xyz_rlatlon[:, 5]), lmax=l_max)
    sph_new_r = shtools.expand.MakeGridPoint(ss_clm, np.rad2deg(sph_xyz_rlatlon[:, 4]),
                                             np.rad2deg(sph_xyz_rlatlon[:, 5]))
    sph.points = (sph.points.T * sph_new_r / sph_xyz_rlatlon[:, 3]).T

    sph_xyz_rlatlon[:, 3] = sph_new_r

    sph_b = sph.sample(mesh_b)

    normal_dot_bvec = np.sum(sph_b.point_normals * sph_b['Bxyz'], axis=1)
    print(np.shape(normal_dot_bvec))
    cos_rvec_normal = np.sum(sph.points * sph_b.point_normals, axis=1) / sph_xyz_rlatlon[:, 3]
    print(cos_rvec_normal)
    dflux = normal_dot_bvec * sph_xyz_rlatlon[:, 3] ** 2 * np.cos(sph_xyz_rlatlon[:, 4]) / cos_rvec_normal
    print(np.shape(dflux))
    flux = np.nansum(dflux) * (2 * np.pi / 180) * (np.pi / 180) * 1e-9
    return flux


if __name__ == '__main__':
    path_Result = 'RESULT/'
    path_2D = 'MESH/2D/'
    name_Result = '((SphR3Ref2-SphR1Ref2_Ref0)dip_SS-SphR1Ref2_Ref0)dip'  # '((SphR3Ref2-SphR1Ref2_Ref0)c2158_SS-SphR1Ref2_Ref0)c2158'
    exportname = name_Result + '_SS'
    name_ss = '(SphR3Ref1-SphR1Ref1_Ref1)dip_SS'  # '(SphR3Ref2-SphR1Ref2_Ref0)c2158_SS'
    magflux = magflux_on_ss(name_Result, name_ss, )

    # mesh_b = pyvista.read(path_Result + name_Result + '_result.vtk')
    # source_surface_extractor(mesh_b, ss_Btot=-1, Rss=3., smooth_lmax=10, path_2D=path_2D, sph_ini_path='SphR3Ref1',
    #                          exportname=exportname)
    # br_ss, clm_ss = br_on_ss(name_Result, exportname, )
