from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

import pickle

import os

import time
import torch

from eval_utils import Dataset, Results, arguments, do_evaluations_slurm, DEFAULT_SEED, HERE, METHODS, METRICS, eval_method, set_seed
from tabpfn.scripts.tabular_metrics import (calculate_score, time_metric)

def post_process_chunks_result(args, result):
    final_results = {}
    for key in result:
        final_results[key] = []
        new_item = {}
        sum_aggregate_metric = torch.tensor(0.0)
        for i, item in enumerate(result[key]):
            individual_result = item.result() if args.slurm else item
            sum_aggregate_metric += individual_result['sum_aggregate_metric']
            new_item = {**new_item, **individual_result}
        new_item.pop('sum_aggregate_metric', None)
        new_item['mean_metric'] = sum_aggregate_metric / ((i+1)*args.chunk_size)
        final_results[key] = new_item

    return final_results

if __name__ == "__main__":
    args = arguments()

    print(args)

    if not args.validation_datasets:
        args.validation_datasets = "cc_valid"
    elif len(args.validation_datasets) == 1 and args.validation_datasets[-1] < 0:
        args.validation_datasets = None

    if not args.test_datasets:
        args.test_datasets = "cc_test"
    elif len(args.test_datasets) == 1 and args.test_datasets[-1] < 0:
        args.test_datasets = None

    # We need to create some directories for this to work
    out_dir = os.path.join(args.result_path, "results", "tabular", "multiclass", f"{time.time()}")
    os.makedirs(out_dir, exist_ok=True
    )

    # We ignore the flags datasets
    filter_f = lambda d: d.name != "flags"  # noqa: ignore

    valid_datasets = []
    test_datasets = []
    if args.validation_datasets is not None:
        valid_datasets = Dataset.fetch(args.validation_datasets, only=filter_f)
    if args.test_datasets is not None:
        test_datasets = Dataset.fetch(args.test_datasets, only=filter_f)

    all_datasets = valid_datasets + test_datasets
    all_datasets = all_datasets
    # base_path = os.path.join('/work/dlclarge1/rkohli-results_tabpfn_180/results_1667931216')

    # print(args.result_path)
    if not args.load_predefined_results:
        result = do_evaluations_slurm(args, all_datasets, slurm=args.slurm, chunk_size=args.chunk_size)
    else:

        def read(_path: Path) -> dict:
            with _path.open("rb") as f:
                return pickle.load(f)

        d = {
            path.stem: read(path)
            for path in args.predefined_results_path.iterdir()
            if path.is_file()
        }
        result = Results.from_dict(
            d,
            datasets=all_datasets,
            recorded_metrics=args.recorded_metrics,
        )

    # Post processing as the results are currently Dict[key, List[Dict]] make them Dict[key, Dict]
    final_results = post_process_chunks_result(args, result)

    datasets_as_lists = [d.as_list() for d in all_datasets]

    # This will update the results in place
    for metric in args.recorded_metrics:
        metric_f = METRICS[metric]
        calculate_score(
            metric=metric_f,
            name=metric,
            global_results=final_results,
            ds=datasets_as_lists,
            eval_positions=args.eval_positions,
        )

    # We also get the times
    calculate_score(
        metric=time_metric,
        name="time",
        global_results=final_results,
        ds=datasets_as_lists,
        eval_positions=args.eval_positions,
    )
    final_results = Results.from_dict(
            final_results,
            datasets=all_datasets,
            recorded_metrics=args.recorded_metrics + ["time"],
        )
    final_results.df.to_csv(os.path.join(out_dir, "results.csv"), index=True)
    # result.df.to_csv(os.path.join(args.result_path, "results.csv"), index=True)