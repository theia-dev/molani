import argparse
import os
import platform
from multiprocessing import cpu_count

from mpi4py import MPI

from . import prepare, manager, worker


def main():
    """
    start the automated molecular animator
    """

    parser = argparse.ArgumentParser(description='molani - automated molecular animator\n'
                                                 '  https://github.com/theia-dev/molani')
    parser.add_argument("config", help="json animation config", type=str)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    render_nodes = comm.Get_size()

    my_rank = MPI.COMM_WORLD.Get_rank()

    config_path = args.config.strip()
    config = prepare.read_config(config_path)

    if my_rank == 0:
        import logging
        logging.basicConfig(format='{asctime} {levelname:>8s} | {message}', style='{', level=logging.INFO)
        log_format = logging.Formatter('{asctime} {levelname:>8s} | {message}', style='{')
        logger = logging.getLogger()
        if 'log_file' in config:
            fh = logging.FileHandler(config['log_file'])
        else:
            log_file_path = str((config['analysis_dir'] / config['movie_name']).with_suffix('.log').absolute())
            fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        logger.addHandler(fh)
        logging.info('#####################')
        logging.info(f'System is starting up on {render_nodes} nodes')

    try:
        hostname = os.environ['SLURMD_NODENAME']
    except KeyError:
        hostname = platform.node()

    all_hostnames = comm.allgather((my_rank, hostname))

    if 'cpu_per_task' in config:
        my_cpu = config['cpu_per_task']
    else:
        try:
            my_cpu = int(os.environ['SLURM_CPUS_PER_TASK'])
        except (KeyError, ValueError):
            my_cpu = cpu_count()
    all_cpus = comm.allgather((my_rank, my_cpu))

    config['my_rank'] = my_rank
    config['cpu_map'] = dict(all_cpus)
    config['cpu_count'] = sum(config['cpu_map'].values())
    config['nodes'] = render_nodes
    config['node_names'] = dict(all_hostnames)
    config['workers'] = list(set([rank for rank, hn in all_hostnames]) - {0})

    if my_rank == 0:
        for node_str in [f"\t{rank}: {hn} ({config['cpu_map'][rank]} cpus)" for rank, hn in all_hostnames]:
            logging.info(node_str)
        manager.manage(config, comm)
    else:
        worker.work(config, comm)

    if my_rank == 0:
        logger.info(f'Script shutting down')
