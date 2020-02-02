import json
from pathlib import Path

import numpy as np


def read_config(config_file):
    """
    Import config

    Parameters
    ----------
    config_file

    Returns
    -------
    dict
        loaded configuration file
    """
    base_file = Path(__file__).absolute().parent / 'base_config.json'
    base_config = json.load(base_file.open())
    task_file = Path(config_file)
    task_config = json.load(task_file.open())
    config = {**base_config, **task_config}
    analysis_dir = (Path(config['base_dir']) / config['name']).absolute()
    config['analysis_dir'] = analysis_dir
    movie_folder = config['analysis_dir'] / config['movie_name']
    config['movie_folder'] = movie_folder.absolute()
    config['movie_raw_folder'] = (config['movie_folder'] / 'raw').absolute()
    config['movie_info_folder'] = (config['movie_folder'] / 'info').absolute()
    config['movie_annotated_folder'] = (config['movie_folder'] / 'annotated').absolute()
    config['movie_cropped_folder'] = (config['movie_folder'] / 'cropped').absolute()
    return config


def read_status(config):
    status_file = config['movie_dir'] / 'status.json'
    if status_file.is_file():
        return json.load(status_file.open())
    else:
        return None


def setup_system(config):
    """
    Created needed folders

    Parameters
    ----------
    config: dict
        loaded configuration file

    """
    config['movie_raw_folder'].mkdir(exist_ok=True, parents=True)
    config['movie_annotated_folder'].mkdir(exist_ok=True, parents=True)
    config['movie_info_folder'].mkdir(exist_ok=True, parents=True)
    config['movie_cropped_folder'].mkdir(exist_ok=True, parents=True)


def supplement_config(config):
    """
    supplement config with additional information

    Parameters
    ----------
    config: dict
        loaded configuration file

    """
    config['frame_raw_name'] = str((config['movie_raw_folder'] / (config['movie_name'] + '_{frame:06}.tga')).absolute())
    config['frame_cropped_name'] = str((config['movie_cropped_folder'] / (config['movie_name'] + '_{frame:06}.png')).absolute())
    config['frame_annotated_name'] = str((config['movie_annotated_folder'] / (config['movie_name'] + '_{frame:06}.jpg')).absolute())
    config['frame_ffmpeg'] = str((config['movie_annotated_folder'] / (config['movie_name'] + '_%06d.jpg')).absolute())
    config['movie_file'] = str((config['movie_folder'] / (config['movie_name'])).absolute().with_suffix('.mp4'))

    config['restart_file'] = config['movie_folder'] / 'restart.json'
    config['performance_file'] = config['movie_info_folder'] / 'performance.json'
    config['full_config_file'] = config['movie_info_folder'] / 'full_config.json'

    universe_meta = json.load((config['analysis_dir'] / config['trajectory']).with_suffix('.json').open())
    if config['type'] in ['cluster_size', 'cluster_set', 'cluster_set_pc']:
        scene_meta = json.load((config['analysis_dir'] / config['meta_data']).with_suffix('.json').open())
        config['scene_meta'] = scene_meta

    if config['type'].startswith('cluster_size'):
        sizes = [c['size'] for c in sum([list(ts[1].values()) for ts in config['scene_meta']], [])]
        config['scale_min'] = min(sizes)
        config['scale_max'] = max(sizes)
    elif config['type'].startswith('cluster_set'):
        config['scale_min'] = -1
        config['scale_max'] = max([c['id'] for c in config['scene_meta'].values()])+1
    config['universe_meta'] = universe_meta
    config['vmd_scale'] = 3 / (universe_meta['dy'] * config['render_scale'])
    config['height_nm'] = universe_meta['dy'] * config['final_scale'] / 10.0
    if config['final_scale'] != config['render_scale']:
        config['height'] = int(np.ceil(config['height'] / config['final_scale'] * config['render_scale'] / 2.0) * 2.0)
    config['width'] = int(np.ceil(config['height'] * universe_meta['dx'] / universe_meta['dy'] / 2.0) * 2.0)

    if 'rotations_per_minute' in config:
        config['rotate_step'] = (360.0 * config['rotations_per_minute']) / (config['fps'] * 60.0)
    else:
        config['rotate_step'] = 0.0

    return config
