import json
import os
import shutil
import tempfile
import time
from multiprocessing import Process, Pipe
from pathlib import Path

import arrow

from . import prepare, worker


def manage(config, comm):
    """
    logic to manage the render sequence

    Parameters
    ----------
    config: dict
        loaded configuration file
    comm: mpi4py.MPI.Intracomm
        MPI inter communicator

    """
    import logging
    work_started = arrow.utcnow()
    logger = logging.getLogger()
    logger.info('Switch to manager mode')
    my_rank = config['my_rank']
    prepare.setup_system(config)
    config = prepare.supplement_config(config)
    full_frame_count = len(config['universe_meta']['timestamp'])
    task_pending = {}
    task_todo = list(range(full_frame_count))
    if config['restart_file'].is_file():
        task_done = set(json.load(config['restart_file'].open()))
        task_todo = sorted(list(set(task_todo)-task_done))
    else:
        task_done = set()

    frame_count = len(task_todo)
    if frame_count == 0:
        frame_count = full_frame_count
    logger.info(f'Prepare to render {frame_count} frames')

    performance = []
    local_worker = None
    request_list = {}
    done_list = {}
    for w_rank in config['workers']:
        request_list[w_rank] = {'ready': comm.irecv(source=w_rank, tag=100),
                                'send': comm.irecv(source=w_rank, tag=200),
                                'done': comm.irecv(source=w_rank, tag=300)}
        done_list[w_rank] = False
    done_list[0] = False

    # Prepare local worker
    os.environ["VMDFORCECPUCOUNT"] = str(config['cpu_map'][my_rank])
    work_dir = tempfile.TemporaryDirectory(prefix=f'mpi_render_{my_rank:05}_')
    work_dir_path = Path(work_dir.name).absolute()
    config['work_dir'] = work_dir_path
    worker.generate_vmd_colormap(config, file_path=config['work_dir'] / 'color_map.tcl')
    save_config = {}
    [save_config.update({str(key): str(value)}) for key, value in config.items()]
    json.dump(save_config, config['full_config_file'].open('w'), sort_keys=True, indent=4)
    # Main loop
    report_stats = max(config['log_report_frequency'],
                       len(task_done) / frame_count * 100 + config['log_report_frequency'] / 2)
    while True:
        if all(done_list.values()):
            break
        if len(task_done)/full_frame_count*100 > report_stats:
            report_stats = max(report_stats+config['log_report_frequency'],
                               len(task_done)/frame_count*100+config['log_report_frequency']/2)
            logger.info(f'######## Status Report ########')
            logger.info(f'Work started {work_started.humanize()}')
            logger.info(f'Done {len(task_done)} of {full_frame_count} frames')
            finish_seconds = (arrow.utcnow()-work_started).seconds/len(task_done)*frame_count
            logger.info(f'Finishing {work_started.shift(seconds=finish_seconds).humanize()}')
            logger.info(f'###############################')
            json.dump(performance, config['performance_file'].open('w'), sort_keys=True, indent=4)

        # Manage remote workers
        for w_rank in config['workers']:
            if not done_list[w_rank]:
                result = request_list[w_rank]['done'].test()
                if result[0]:
                    if result[1] == 'done':
                        done_list[w_rank] = True
            else:
                continue
            result = request_list[w_rank]['send'].test()
            if result[0]:
                first, last = result[1]
                logger.info(f'Worker {config["node_names"][w_rank]} ({w_rank}) finished frame set [{first}, {last}]')

                done_task_list = list(range(first, last+1))

                assert task_pending[w_rank][0] == done_task_list
                task_done.update(done_task_list)
                time_taken = task_pending[w_rank][1]
                task_pending.__delitem__(w_rank)
                request_list[w_rank]['send'] = comm.irecv(source=w_rank, tag=200)
                comm.send('saved', dest=w_rank, tag=210)
                performance.append((len(done_task_list), (arrow.utcnow()-time_taken).seconds,
                                    config['cpu_map'][w_rank], w_rank))
                json.dump(list(task_done), config['restart_file'].open('w'), sort_keys=True, indent=4)
                logger.info(f'Frame set [{first}, {last}] done in {time_taken.humanize(only_distance=True)} '
                            f'from worker {config["node_names"][w_rank]} ({w_rank})')

            result = request_list[w_rank]['ready'].test()
            if result[0]:
                if result[1][0] == 'ready':
                    if w_rank in task_pending:
                        logger.error(f'Lost results from {config["node_names"][w_rank]} ({w_rank})')
                        exit(f'Lost results from {config["node_names"][w_rank]} ({w_rank})')

                    task_list = task_todo[:config['chunk_size']*config['cpu_map'][w_rank]]
                    if task_list:
                        task = [min(task_list), max(task_list)]
                    else:
                        task = []
                    logger.info(f'Try sending frame set {str(task)} to worker {config["node_names"][w_rank]} ({w_rank})')
                    comm.send(task, dest=w_rank, tag=110)
                    task_pending[w_rank] = [task_list, arrow.utcnow()]
                    [task_todo.remove(task) for task in task_list]
                    logger.info(f'Successful send frame set {str(task)} to worker {config["node_names"][w_rank]} ({w_rank})')
                request_list[w_rank]['ready'] = comm.irecv(source=w_rank, tag=100)

        # Manage locale worker
        if local_worker is None:
            if 0 in task_pending:
                logger.error(f"Lost results from local worker")
                exit(f"Lost results from local worker")
            task_list = task_todo[:config['chunk_size'] * config['cpu_map'][0]]
            if task_list:
                task = [min(task_list), max(task_list)]
                local_manager_conn, local_worker_conn = Pipe()
                logger.info(f'Try sending frame set {str(task)} to local worker')
                local_worker = Process(target=worker.local_work, args=(config, task, local_worker_conn))
                local_worker.start()
                task_pending[0] = [task_list, arrow.utcnow()]
                [task_todo.remove(task) for task in task_list]
                logger.info(f'Successful send frame set {str(task)} to local worker')
            else:
                done_list[0] = True
        else:
            local_worker_ec = local_worker.exitcode
            if local_worker_ec is not None:
                if local_worker_ec < 0:
                    logger.error(f"Local worker died ({local_worker_ec})")
                    exit(f"Local worker died ({local_worker_ec})")

                first, last = local_manager_conn.recv()
                logger.info(f'Local worker finished frame set [{first}, {last}]')
                done_task_list = list(range(first, last + 1))
                assert task_pending[0][0] == done_task_list
                task_done.update(done_task_list)
                time_taken = task_pending[0][1]
                task_pending.__delitem__(0)
                performance.append((len(done_task_list), (arrow.utcnow() - time_taken).seconds,
                                    config['cpu_count'], 0))
                json.dump(list(task_done), config['restart_file'].open('w'), sort_keys=True, indent=4)
                logger.info(f'Frame set [{first}, {last}] done in {time_taken.humanize(only_distance=True)} from local worker')
                local_worker = None
        time.sleep(1)

    work_dir.cleanup()
    logger.info(f'Done rendering {len(task_done)} frames')

    logger.info(f'Start to finalize the movie on {config["cpu_map"][my_rank]} cpus')
    worker.make_movie(config)

    logger.info(f'Movie saved under {config["movie_file"]}')

    config['restart_file'].unlink()
    if config['clean_up']:
        logger.info(f'Clean up rendered images')
        shutil.rmtree(config['movie_raw_folder'])
        shutil.rmtree(config['movie_annotated_folder'])
        shutil.rmtree(config['movie_cropped_folder'])





