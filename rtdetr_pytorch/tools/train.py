"""by lyuwenyu
"""

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# import os
# import sys
import argparse
from pprint import pprint

import rtdetr.misc.dist as dist
from rtdetr.core import YAMLConfig
from rtdetr.solver import DetSolver


def main(
    args,
) -> None:
    """main"""
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    cfg = YAMLConfig(
        args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning
    )
    pprint(cfg.yaml_cfg)
    solver = DetSolver(cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--tuning",
        "-t",
        type=str,
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed",
    )
    args = parser.parse_args()

    main(args)
