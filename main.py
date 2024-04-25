import concurrent.futures
import datetime
import os
import time

from src.model.algorithm import algorithm_body
from src.model.parse_arguments import get_arguments


def single_fold_run(para_fold):
    console_args = get_arguments()
    print(console_args)
    project_folder = os.path.dirname(__file__)
    ablation = '' if console_args.e2 & console_args.e3 & console_args.e4 else 'ablation'
    control_args = {
        'name': 'DWHRec',
        'dataset': {'name': console_args.dataset, 'fold': para_fold, },
        'model': {'hypergraph': console_args.ph, 'walk': console_args.pw, 'embedding': console_args.pe,
                  'recommendation': console_args.pr},
        'ablation': {'tag-track': console_args.e2, 'album-track': console_args.e3, 'artist-track': console_args.e4},
        'parameter': {'r': console_args.r, 'k': console_args.k, 's': console_args.s, 'w': console_args.w,
                      'n': console_args.n, },
        'path': {'data_folder': os.path.join(project_folder, 'datasets', console_args.dataset),
                 'temp_folder': os.path.join(project_folder, 'temp', ablation, console_args.dataset, ','.join(
                     ['r=' + str(console_args.r), 'k=' + str(console_args.k), 's=' + str(console_args.s),
                      'w=' + str(console_args.w), 'e2=' + str(console_args.e2), 'e3=' + str(console_args.e3),
                      'e4=' + str(console_args.e4)])),
                 'result_folder': os.path.join(project_folder, 'results', ablation, console_args.dataset, ','.join(
                     ['r=' + str(console_args.r), 'k=' + str(console_args.k), 's=' + str(console_args.s),
                      'w=' + str(console_args.w), 'e2=' + str(console_args.e2), 'e3=' + str(console_args.e3),
                      'e4=' + str(console_args.e4)])), },
    }
    print('Control arguments:', control_args)
    model_start_time = datetime.datetime.now()
    print(' '.join(['Fold:', str(para_fold), 'start at:', str(model_start_time)]))
    # body
    algorithm_body(control_args)
    model_finish_time = datetime.datetime.now()
    elapsed_time = (model_finish_time - model_start_time).seconds / 3600
    print(' '.join(['Fold:', str(para_fold), 'finished, elapsed time:', str(elapsed_time), 'h.']))


if __name__ == '__main__':
    folds_array = list(range(10))
    start_time = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = list(executor.submit(single_fold_run, fold) for fold in folds_array)
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    end_time = time.time()
    print('Finished, total time:', str((end_time - start_time) / 3600), 'h.')
