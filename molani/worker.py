import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import command_lib
from . import prepare


def generate_vmd_colormap(config, file_path=None, cm_min=0, cm_max=1):
    """
    Generate tcl script snippet to create a colormap in VMD.

    Parameters
    ----------
    config: dict
        loaded configuration file
    file_path: Path or str
        save path for tcl script snippet
    cm_min: float or int
        min value for the colormap
    cm_max: float or int
        max value for the colormap

    """
    fill_cmap = plt.get_cmap('bone')
    shift = []
    if config:
        if config['type'] == 'solid_color':
            return
        cmap_list = []
        for selection in config['color_map']:
            if type(selection[1]) is str:
                cmap_list.append(plt.get_cmap(selection[1]))
                shift.append(0)
            elif type(selection[1]) is list:
                cmap_list.append(plt.get_cmap(selection[1][0]))
                shift.append(selection[1][1])
    else:
        cmap_list = [plt.get_cmap('plasma')]

    size = 1024
    offset = 33

    size_per_cmap = size // len(cmap_list)

    color_list = []
    for n, cmap in enumerate(cmap_list):
        color_list.append([(i+size_per_cmap*n, np.array(cmap(ci))*(1.0+shift[n])) for i, ci in enumerate(np.linspace(cm_min, cm_max, size_per_cmap))])
    color_list.append([(i + size_per_cmap * len(cmap_list), fill_cmap(ci)) for i, ci in enumerate(np.linspace(cm_min, cm_max, size % len(cmap_list)))])
    color_list = sum(color_list, [])
    color_list = [f"color change rgb {c[0] + offset} {min(1.0, c[1][0]):6.5F} {min(1.0, c[1][1]):6.5F} {min(1.0, c[1][2]):6.5F}" for c in color_list]

    if config:
        if config['my_rank'] == 0:
            Path(config['movie_info_folder'] / f'colormap_vmd.tcl').write_text("\n".join(color_list))
            for n, cmap in enumerate(cmap_list):
                data = list(np.linspace(cm_min, cm_max, 512))*64
                data = np.array(data).reshape((64, 512))
                norm = plt.Normalize(vmin=data.min(), vmax=data.max())
                # map the normalized data to colors
                # image is now RGBA (512x512x4)

                image = cmap(norm(data))*(1+shift[n])

                # save the image
                plt.imsave(config['movie_info_folder'] / f'colormap_full{n}.png', image)
            if config['type'].startswith('cluster_set'):
                for n, cmap in enumerate(cmap_list):
                    data = []
                    raw_data = list(np.linspace(cm_min, cm_max, config['scale_max']+2)[1:-1])
                    extend = int(512/config['scale_max'])
                    for i in range(config['scale_max']):
                        data.extend([raw_data[i]]*extend)
                    data = data*64
                    data = np.array(data).reshape((64, extend*config['scale_max']))

                    norm = plt.Normalize(vmin=0, vmax=1)
                    # map the normalized data to colors
                    # image is now RGBA (512x512x4)

                    image = cmap(norm(data)) * (1 + shift[n])
                    image[:, :, 3] = 1.0
                    # save the image
                    plt.imsave(config['movie_info_folder'] / f'colormap_seq{n}.png', image)

    if file_path:
        Path(file_path).write_text("\n".join(color_list))
        return
    else:
        return color_list


def build_render_script(config, task):
    """
    Generate VMD tcl script to create the scene.

    Parameters
    ----------
    config: dict
        loaded configuration file
    task: Tuple[int, int]
        first and last frame of task

    """
    render_script_path = config['work_dir'] / 'render.tcl'
    render_scrip_text = ""

    render_scrip_text += command_lib.new_mol.format(
        path=(str((config['analysis_dir'] / config['trajectory']).with_suffix('.pdb').absolute())))

    render_scrip_text += command_lib.display_setting.format(
        x=config['height'],
        y=config['width'],
        shadows=config['shadows'],
        depthcue=config['depthcue'],
        antialias=config['antialias'],
        ao=config['ambientocclusion'],
        aoambient=config['aoambient'],
        aodirect=config['aodirect'],
        projection=config['projection'],
        nearclip=config['nearclip'])

    render_scrip_text += command_lib.mol_del_style.format(rid=0)
    render_scrip_text += command_lib.color_background.format(**config['background'])
    render_scrip_text += command_lib.color_foreground.format(**config['foreground'])

    if not config['type'] == 'solid_color':
        render_scrip_text += command_lib.play_script.format(path=str(config['work_dir'] / 'color_map.tcl'))

    if config['type'] == 'cluster_size':
        trj_data = True
    elif config['type'].startswith('cluster_set'):
        trj_data = False
    else:
        trj_data = True

    for rid, entry in enumerate(config['color_map']):

        if config['type'] == 'solid_color':
            render_scrip_text += command_lib.set_color.format(
                id=rid+2,
                r=entry[1][0], g=entry[1][1], b=entry[1][2])
            render_scrip_text += command_lib.mol_style_base.format(
                rid=rid,
                select=entry[0],
                style=config['style'],
                material=config['material'])

            render_scrip_text += command_lib.mol_style_color_ID.format(
                rid=rid,
                cid=rid+2)

            render_scrip_text += command_lib.mol_style_periodic.format(
                rid=rid,
                periodic=config['periodic']
            )

        else:
            render_scrip_text += command_lib.mol_style_base.format(
                rid=rid,
                select=entry[0],
                style=config['style'],
                material=config['material'])

            render_scrip_text += command_lib.mol_style_color_name.format(
                rid=rid,
                name=command_lib.value_display_names[
                    command_lib.value_names[rid] if trj_data else command_lib.value_names_system[rid]],
                min=0,
                max=1023)

            render_scrip_text += command_lib.mol_style_periodic.format(
                rid=rid,
                periodic=config['periodic']
            )

    if config['type'] == 'cluster_set_pc':
        render_scrip_text += command_lib.mol_style_base.format(
            rid=len(config['color_map']),
            select="resname PC",
            style=config['style'],
            material=config['material'])

        render_scrip_text += command_lib.mol_style_color_ID.format(
            rid=len(config['color_map']),
            cid=6)

    render_scrip_text += command_lib.add_trajectory.format(
        path=(str((config['analysis_dir'] / config['trajectory']).with_suffix('.dcd').absolute())),
        first=max(0, task[0]-1),
        last=task[1]-1)

    render_scrip_text += command_lib.scale.format(scale=config['vmd_scale'])

    size_per_cmap = 1024 // len(config['color_map'])

    all_members = set()
    if config['type'] == 'cluster_size':
        ts, frame_meta = config['scene_meta'][0]
        for cluster in frame_meta.values():
            all_members.update(set(cluster['members']))

        for member in all_members:
            render_scrip_text += command_lib.create_selection.format(
                selection_name=member,
                select=f'resname {member}')
    elif config['type'].startswith('cluster_set'):
        for cluster_name, entry in config['scene_meta'].items():
            for rid, member_list in enumerate(entry['selection']):
                render_scrip_text += command_lib.create_selection.format(
                    selection_name=f'{cluster_name}_{rid}',
                    select=" or ".join([f'resname {member}' for member in member_list]))
                render_scrip_text += command_lib.set_value.format(
                    selection_name=f'{cluster_name}_{rid}',
                    name=command_lib.value_names[rid] if trj_data else command_lib.value_names_system[rid],
                    value=int(np.floor(
                        size_per_cmap * rid + (size_per_cmap - 1) * (entry['id'] - config['scale_min']) / (
                                    config['scale_max'] - config['scale_min']))))

    for i, frame in enumerate(range(task[0], task[1]+1)):
        render_scrip_text += command_lib.load_frame.format(frame=i+1)
        render_scrip_text += command_lib.rotate_to.format(
            axis=config['rotate_axis'],
            angle=config['rotate_step'] * frame)

        if config['type'] == 'cluster_size':
            ts, frame_meta = config['scene_meta'][frame]
            for cluster in frame_meta.values():
                for rid in range(len(config['color_map'])):
                    for member in cluster['member']:
                        render_scrip_text += command_lib.set_value.format(
                            selection_name=member,
                            name=command_lib.value_names[rid],
                            value=int(np.floor(size_per_cmap*rid+(size_per_cmap-1)*(cluster['size']-config['scale_min'])/(config['scale_max']-config['scale_min']))))

        render_scrip_text += command_lib.render.format(
            temp_path=config['work_dir'] / 'render_scene.dat',
            tachyon=config['tachyon'],
            path=config['frame_raw_name'].format(frame=frame),
            cpu=config['cpu_map'][config['my_rank']])

    render_scrip_text += command_lib.vmd_exit

    render_script_path.write_text(render_scrip_text)
    return render_scrip_text


def crop_frame(data):
    """
    Crop frames with convert from imagemagick

    Parameters
    ----------
    data: Tuple(int, list)
        frame number, frame_data

    """
    frame, frame_data = data
    in_path, out_path, change = frame_data
    command = ['convert', in_path.format(frame=frame),
               '-gravity', 'center',
               '-crop', f'{change}%x+0+0',
               '+repage',
               out_path.format(frame=frame)]
    subprocess.run(command)
    return True


def annotate_frame(data):
    """
    Annotate frames with convert from imagemagick

    Parameters
    ----------
    data: Tuple(int, list, float)
        frame number, frame_data, time step (ns)

    """
    frame, frame_data, timestep = data
    in_path, out_path, font, font_size, line_width, axis_select, af, ab, ls, fh, lh, le = frame_data
    command = ['convert', in_path.format(frame=frame),
               '-stroke', ab, '-strokewidth', str(line_width * 3.5), '-draw',
               f'line {ls},{lh} {le},{lh}',
               '-stroke', af, '-strokewidth', str(line_width), '-draw',
               f'line {ls},{lh} {le},{lh}',
               '-pointsize', str(font_size), '-gravity', 'SouthEast',
               '-fill', af, '-stroke', ab, '-strokewidth', str(font_size / 5),
               '-draw', f'font {font} text {ls},{fh} \'{timestep:6.2f} ns\'',
               '-stroke', 'none',
               '-draw', f'font {font} text {ls},{fh} \'{timestep:6.2f} ns\'',
               '-fill', af, '-stroke', ab, '-strokewidth', str(font_size / 4),
               '-gravity', 'SouthWest',
               '-draw', f'font {font} text {ls},{fh} \'{axis_select:.2f} nm\'',
               '-stroke', 'none',
               '-draw', f'font {font} text {ls},{fh} \'{axis_select:.2f} nm\'',
               out_path.format(frame=frame)]
    subprocess.run(command)
    return True


def annotate_movie(config, task):
    """
    Prepare cropping and annotating of the rendered images.

    Parameters
    ----------
    config: dict
        loaded configuration file
    task: Tuple[int, int]
        first and last frame of task

    """
    if config['render_scale'] != config['final_scale']:
        out_path = config['frame_cropped_name']
        in_path = config['frame_raw_name']
        change = 100.0 * config['final_scale'] / config['render_scale']

        frame_data = in_path, out_path, change
        task_list = [(frame, frame_data) for frame in
                     range(task[0], task[1] + 1)]
        with Pool(config['cpu_map'][config['my_rank']]) as task_pool:
            result = task_pool.map(crop_frame, task_list)
        assert all(result)

    if config['render_scale'] != config['final_scale']:
        in_path = config['frame_cropped_name']
    else:
        in_path = config['frame_raw_name']

    im = Image.open(in_path.format(frame=task[0]))
    width, height = im.size
    resolution = height / config['height_nm']
    valid_axis = np.array([0.05, 0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 4, 5, 7.5, 8,
                           10, 12, 14, 15, 16, 18, 20, 25, 30, 35, 40, 45, 50])
    axis_select = valid_axis[np.argmin(np.abs(resolution * valid_axis - width*0.3))]
    line_length = axis_select * resolution
    line_width = max(1, 0.004 * config['height'])
    font_size = int(max(16, 0.05 * config['height']))
    ls = max(10, int(0.02 * config['height']))
    fh = max(10, int(0.014 * config['height']))
    lh = int(round(height - ls + line_width))
    le = int(round(ls + line_length))
    line_width = max(1, 0.004 * config['height'])

    af = config['annotate_foreground']
    ab = config['annotate_background']
    out_path = config['frame_annotated_name']

    font = config['font']
    frame_data = [in_path, out_path, font, font_size, line_width, axis_select, af, ab, ls, fh, lh, le]
    task_list = [(frame, frame_data, config['universe_meta']['timestamp'][frame] / 1000) for frame in range(task[0], task[1]+1)]
    with Pool(config['cpu_map'][config['my_rank']]) as task_pool:
        result = task_pool.map(annotate_frame, task_list)
    assert all(result)


def make_movie(config):
    """
    Combine images with ffmpeg

    Parameters
    ----------
    config: dict
        loaded configuration file

    """
    render_command = ['ffmpeg', '-r', str(config["fps"]), '-i', config['frame_ffmpeg'], '-y',
                      '-c:v', 'libx264', '-tune', 'animation', '-preset', 'slow', '-crf', '20',
                      '-vf', f'fps={config["fps"]}', '-pix_fmt', 'yuv420p',
                      config['movie_file']]

    subprocess.run(render_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def work(config, comm):
    """
    Render images and annotate them externally

    Parameters
    ----------
    config: dict
        loaded configuration file
    comm: mpi4py.MPI.Intracomm
        MPI inter communicator

    """
    my_rank = config["my_rank"]
    os.environ["VMDFORCECPUCOUNT"] = str(config['cpu_map'][my_rank])

    work_dir = tempfile.TemporaryDirectory(prefix=f'mpi_render_{my_rank:05}_')
    work_dir_path = Path(work_dir.name).absolute()
    config['work_dir'] = work_dir_path
    config = prepare.supplement_config(config)
    generate_vmd_colormap(config, file_path=config['work_dir'] / 'color_map.tcl')

    # Main work loop
    while True:
        comm.isend(('ready', my_rank), dest=0, tag=100)
        task_list = comm.recv(source=0, tag=110)
        if task_list:
            render_script = build_render_script(config, task_list)
            subprocess.run([config['vmd_path'], "-dispdev", "text"],
                           input=render_script,
                           encoding='utf8', stdout=subprocess.DEVNULL)
            annotate_movie(config, task_list)

        else:
            comm.send('done', dest=0, tag=300)
            break
        comm.isend(task_list, dest=0, tag=200)
        comm.recv(source=0, tag=210)
    work_dir.cleanup()


def local_work(config, task_list, local_worker_conn):
    """
    Render images and annotate them locally

    Parameters
    ----------
    config: dict
        loaded configuration file
    task_list: Tuple[int, int]
        first and last frame of task
    local_worker_conn: Tuple[connection.Connection, connection.Connection]
        multiprocessing.Pipe

    """
    if task_list:
        render_script = build_render_script(config, task_list)
        Path(config['movie_folder']/'vmd_script.txt').write_text(render_script)
        subprocess.run([config['vmd_path'], "-dispdev", "text"],
                       input=render_script,
                       encoding='utf8', stdout=subprocess.DEVNULL)
        annotate_movie(config, task_list)
        local_worker_conn.send(task_list)
