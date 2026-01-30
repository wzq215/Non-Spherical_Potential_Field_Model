# %%
import os

import numpy as np

from result_combiner import combine_in_out, combine_in_out_trace_from_photosphere, trace_from_photosphere_circle
import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
import pyvista as pv
from datetime import datetime,timedelta
from sunpy.map.maputils import all_coordinates_from_map

pv.global_theme.allow_empty_mesh = True

def convert_to_new_frame(mesh_, map_):
    point_coord = SkyCoord(mesh_.points * u.R_sun,
                           obstime=map_.observer_coordinate.obstime,
                           frame='heliographic_carrington',
                           observer='sun', rsun=map_.coordinate_frame.rsun,
                           representation_type='cartesian')
    point_coord_helioprojective = point_coord.transform_to(frames.HeliocentricInertial(obstime=map_.observer_coordinate.obstime))
    mesh_.points[:,0] = point_coord_helioprojective.represent_as('cartesian').x.value
    mesh_.points[:,1] = point_coord_helioprojective.represent_as('cartesian').y.value
    mesh_.points[:,2] = point_coord_helioprojective.represent_as('cartesian').z.value
    return mesh_


def mask_lasco_c2(lasco_map):
    pixel_coords = all_coordinates_from_map(lasco_map)
    solar_center = SkyCoord(0 * u.deg, 0 * u.deg, frame=lasco_map.coordinate_frame)
    pixel_radii = np.sqrt((pixel_coords.Tx - solar_center.Tx) ** 2 +
                          (pixel_coords.Ty - solar_center.Ty) ** 2)
    # Note that the inner mask extends just beyond 2 solar radii to mask the
    # Fresnel diffraction caused by the occulter edge.
    mask_inner = pixel_radii < lasco_map.rsun_obs * 2.4
    mask_outer = pixel_radii > lasco_map.rsun_obs * 6
    final_mask = mask_inner + mask_outer
    # masked_lasco = sunpy.map.Map(lasco_map.data, lasco_map.meta, mask=final_mask)
    lasco_map.data[final_mask]=0.
    occult_colormap = lasco_map.cmap.copy()
    occult_colormap.set_bad('black')
    return lasco_map, occult_colormap

if __name__ == '__main__':

    magmap_name_list = ['mrzqs240331t2104c2282_050',
                        'mrzqs240331t0304c2282_060',
                        'mrzqs240330t0914c2282_070',
                        'mrzqs240329t1514c2282_080',
                        'mrzqs240328t2004c2282_090',
                        'mrzqs240328t0204c2282_100']

    datetime_lst = [datetime(2024, 3, 28, 2, 4),
                    datetime(2024, 3, 28, 20, 4),
                    datetime(2024, 3, 29, 15, 14),
                    datetime(2024, 3, 30, 9, 14),
                    datetime(2024, 3, 31, 3, 4),
                    datetime(2024, 3, 31, 21, 4), ]
    for i_result in [0]:
        tag_tmp = "{:03}".format((10-i_result) * 10)
        magmap_dt = datetime_lst[i_result]
        magmap_name = 'mrzqs' + magmap_dt.strftime('%Y%m%dt%H%M')[2:] + 'c2282_' + tag_tmp + '.fits.gz'
        magmap_tmp = sunpy.map.Map('E19/magnetogram/' + magmap_name)

        search_begin_dt = magmap_dt - timedelta(minutes=30)
        search_end_dt = magmap_dt +timedelta(minutes=30)
        time_range = a.Time(search_begin_dt.strftime('%Y/%m/%d %H:%M'), search_end_dt.strftime('%Y/%m/%d %H:%M'))

        # 搜索LASCO C2数据
        query = Fido.search(time_range,
                           a.Instrument('LASCO'),
                           a.Detector('C2'))
        # 指定缓存目录（可自定义）
        cache_dir = '/Users/ephe/sunpy/data/'

        # 检查本地是否已有相应文件
        downloaded_files = Fido.fetch(query, path=os.path.join(cache_dir, '{file}'), download=False, overwrite=False)

        # 如果文件不存在，则下载
        missing_files = [f for f in downloaded_files if not os.path.exists(f)]
        if missing_files:
            print(f"检测到 {len(missing_files)} 个缺失文件，正在下载...")
            downloaded_files = Fido.fetch(query, path=os.path.join(cache_dir, '{file}'))
        else:
            print("所有文件已存在，直接读取。")
        # downloaded_files = Fido.fetch(query)
        # print(query)
        lasco_c2_map = sunpy.map.Map(downloaded_files[0])
        corona_map,corona_cmap = mask_lasco_c2(lasco_c2_map)


        # %%
        search_begin_dt = magmap_dt - timedelta(seconds=30)
        search_end_dt = magmap_dt + timedelta(seconds=30)
        time_range = a.Time(search_begin_dt.strftime('%Y/%m/%d %H:%M'), search_end_dt.strftime('%Y/%m/%d %H:%M'))
        query = Fido.search(time_range,
                            a.Instrument('AIA'),
                            # a.Detector('EIT'),
                            a.Wavelength(171 * u.Angstrom)
                            )
        print(query)
        # downloaded_files = Fido.fetch(query)
        # print(downloaded_files)

        # 指定缓存目录（可自定义）
        cache_dir = '/Users/ephe/sunpy/data/'

        # 检查本地是否已有相应文件
        # downloaded_files = Fido.fetch(query, path=os.path.join(cache_dir, '{file}'), download=False, overwrite=False)
        # print(downloaded_files)

        # # 如果文件不存在，则下载
        # missing_files = [f for f in downloaded_files if not os.path.exists(f)]
        # if missing_files:
        #     print(f"检测到 {len(missing_files)} 个缺失文件，正在下载...")
        #     downloaded_files = Fido.fetch(query, path=os.path.join(cache_dir, '{file}'))
        # else:
        #     print("所有文件已存在，直接读取。")

        downloaded_files = [cache_dir+'aia.lev1.171A_2024_03_28T02_03_33.35Z.image_lev1.fits']
        secchi_map = sunpy.map.Map(downloaded_files[0])


        hi_map = secchi_map

        # %%
        observer_coord = hi_map.observer_coordinate
        observer_coord.observer='sun'
        from sunpy.coordinates import frames
        observer_coord_carrington = observer_coord.transform_to(frames.HeliographicCarrington)
        observer_coord_carrington_x = observer_coord_carrington.represent_as('cartesian').x.value
        observer_coord_carrington_y = observer_coord_carrington.represent_as('cartesian').y.value
        observer_coord_carrington_z = observer_coord_carrington.represent_as('cartesian').z.value

        # %%
        PATH_2D = 'MESH/2D/'
        PATH_3D = 'MESH/3D/'
        PATH_RESULT = 'RESULT/'
        PATH_MAGMAP = 'magnetogram/'

        # i_result=9

        OuterSphere_Rs = 10
        MiddleSphere_Rs_str = '2d2'
        InnerSphere_Rs = 1
        CR_tag='c2282_'+tag_tmp
        SS_tag=''
        Ref_tag = 'Ref3'

        PATH_RESULT = f'RESULT/CR2282_E19/NSPF_Rss{MiddleSphere_Rs_str}_{Ref_tag}/' + tag_tmp + '/'

        NAME_RESULT2 = '(SphR'+str(OuterSphere_Rs)+'Ref3-(SphR'+MiddleSphere_Rs_str+'Ref3-SphR'+str(InnerSphere_Rs)+'Ref3_Ref0)'+CR_tag+'_SS'+SS_tag+'_Ref0)array'
        NAME_RESULT1 = '((SphR'+MiddleSphere_Rs_str+'Ref3-SphR'+str(InnerSphere_Rs)+'Ref3_Ref0)'+CR_tag+'_SS'+SS_tag+'-SphR'+str(InnerSphere_Rs)+'Ref3_Ref0)'+CR_tag
        NAME_SS = '(SphR'+MiddleSphere_Rs_str+'Ref3-SphR'+str(InnerSphere_Rs)+'Ref3_Ref0)'+CR_tag+'_SS'+SS_tag

        fov_normal = [observer_coord_carrington_x, observer_coord_carrington_y, observer_coord_carrington_z]
        circ_polar = np.cross(np.cross(fov_normal, [1, 0, 0]), fov_normal)
        circ_polar = circ_polar / np.linalg.norm(circ_polar) * 1.01

        # %%
        inner_mesh = pv.read(PATH_RESULT + NAME_RESULT1 + '.vtk')
        inner_mesh_slice = inner_mesh.slice(normal=fov_normal, origin=[0,0,0])
        inner_mesh_slice_HCI = convert_to_new_frame(inner_mesh_slice, corona_map)


        # inner_Blines_slice_from_sun, outer_Blines_slice_from_sun = \
        #     trace_from_photosphere_circle(PATH_RESULT, NAME_RESULT1, NAME_RESULT2, PATH_2D, NAME_SS,
        #                                   normal=fov_normal, polar=circ_polar)
        inner_Blines_slice_from_sun, outer_Blines_slice_from_sun,_ = combine_in_out(PATH_RESULT, NAME_RESULT1, NAME_RESULT2,
                                                                                    PATH_2D, NAME_SS,
                                                                                    slice_normal = fov_normal,
                                                                                    slice_origin=[0,0,0])

        # %%
        inner_Blines_slice_from_sun_HCI = convert_to_new_frame(inner_Blines_slice_from_sun, hi_map)
        # outer_Blines_slice_from_sun_HCI = convert_to_new_frame(outer_Blines_slice_from_sun, hi_map)
        outer_Blines_slice_from_sun_HCI = []
        for outer_Bline in outer_Blines_slice_from_sun:
            outer_Blines_slice_from_sun_HCI.append(convert_to_new_frame(outer_Bline, hi_map))

        # %%
        from sunkit_pyvista import SunpyPlotter
   # %%
        observer_coord_HCI = observer_coord.transform_to(frames.HeliocentricInertial)
        observer_coord_HCI_x = observer_coord_HCI.represent_as('cartesian').x.value
        observer_coord_HCI_y = observer_coord_HCI.represent_as('cartesian').y.value
        observer_coord_HCI_z = observer_coord_HCI.represent_as('cartesian').z.value

# %%
        pv.set_plot_theme(pv.themes.DarkTheme())
        p = SunpyPlotter(window_size=(1700, 1500))# [0.3,0.5,0.9,1.0]
        # p.plot_map(corona_map, clip_interval=[1., 99.99] * u.percent,opacity=[0.0, 0.1, 0.2,0.5, 1.0],cmap='gray')
        p.plot_map(corona_map, clip_interval=[10, 99.7] * u.percent,
                   opacity=[0.0, 0.0, 1.0],
                   cmap='gray')
        p.plot_map(hi_map, clip_interval=[10,99]*u.percent,opacity=[0.0,0.9,1.])
        # p.plot_map(outmap_default)
        # p.plotter.add_mesh(inner_Blines_slice_HCI.tube(radius=0.01,),color='green',opacity=1)
        # p.plotter.add_mesh(outer_Blines_slice_HCI.tube(radius=0.01,),color='green',opacity=1)
        p.plotter.add_mesh(inner_mesh_slice_HCI,opacity=0.2,color='lightgreen',log_scale=True)
        inner_Blines_slice_from_sun_HCI.set_active_scalars('Br')
        # p.plotter.add_mesh(inner_Blines_slice_from_sun_HCI.tube(radius=0.005,),cmap='coolwarm',clim=[-10.,10.],opacity=0.3)
        for Bline in outer_Blines_slice_from_sun_HCI:
            if len(Bline.points) > 0:
                Bline.set_active_scalars('Br')
                p.plotter.add_mesh(Bline,cmap='coolwarm',clim=[-.1,.1], opacity=[1.0,0.3,1.0],show_scalar_bar=False)
        # p.plotter.add_mesh(inner_Blines.tube(radius=0.01, ), color='blue', )
        # p.plotter.add_mesh(outer_Blines.tube(radius=0.01, ), color='green', )
        # p.plotter.add_mesh(ss_mesh, opacity=0.5)
        # p.plotter.add_slice_
        # p.plotter.show_grid()
        # p.plotter.add_title(magmap_dt.strftime('%Y/%m/%d %H:%M')+'\nNSPF 2.2Rs')
        Rs2km=696300
        p.plotter.camera_position=np.array([observer_coord_HCI_x,observer_coord_HCI_y,observer_coord_HCI_z])/20.
        # p.plotter.camera_focus=np.array([0.,0.,1.])*Rs2km
        # p.plotter.camera.zoom(8)
        p.show()
        # p.plotter.screenshot('NSPF_2.2_zoomin.png')
        # p.plotter.close()
# %%
        pv.set_plot_theme(pv.themes.DarkTheme())
        p = SunpyPlotter(window_size=(1700, 1700))  # [0.3,0.5,0.9,1.0]
        # p.plot_map(corona_map, clip_interval=[1., 99.99] * u.percent,opacity=[0.0, 0.1, 0.2,0.5, 1.0],cmap='gray')
        p.plot_map(corona_map, clip_interval=[10, 99.7] * u.percent,
                   opacity=[0.0, 0.0, 1.0],
                   cmap='gray')
        p.plot_map(hi_map, clip_interval=[10, 99] * u.percent, opacity=[0.0, 0.9, 1.])
        # p.plot_map(outmap_default)
        # p.plotter.add_mesh(inner_Blines_slice_HCI.tube(radius=0.01,),color='green',opacity=1)
        # p.plotter.add_mesh(outer_Blines_slice_HCI.tube(radius=0.01,),color='green',opacity=1)
        # p.plotter.add_mesh(inner_mesh_slice_HCI, opacity=0.2, color='lightgreen', log_scale=True)
        # inner_Blines_slice_from_sun_HCI.set_active_scalars('Br')
        # # p.plotter.add_mesh(inner_Blines_slice_from_sun_HCI.tube(radius=0.005,),cmap='coolwarm',clim=[-10.,10.],opacity=0.3)
        # for Bline in outer_Blines_slice_from_sun_HCI:
        #     if len(Bline.points) > 0:
        #         Bline.set_active_scalars('Br')
        #         p.plotter.add_mesh(Bline, cmap='coolwarm', clim=[-.1, .1], opacity=[1.0, 0.3, 1.0],
        #                            show_scalar_bar=False)
        # p.plotter.add_mesh(inner_Blines.tube(radius=0.01, ), color='blue', )
        # p.plotter.add_mesh(outer_Blines.tube(radius=0.01, ), color='green', )
        # p.plotter.add_mesh(ss_mesh, opacity=0.5)
        # p.plotter.add_slice_
        # p.plotter.show_grid()
        # p.plotter.add_title(magmap_dt.strftime('%Y/%m/%d %H:%M')+'\nNSPF 2.2Rs')
        Rs2km = 696300
        p.plotter.camera_position = np.array([observer_coord_HCI_x, observer_coord_HCI_y, observer_coord_HCI_z]) / 20.
        # p.plotter.camera_focus=np.array([0.,0.,1.])*Rs2km
        p.plotter.camera.zoom(5)
        p.show()
        p.plotter.screenshot('plain.png')
        # p.plotter.close()
