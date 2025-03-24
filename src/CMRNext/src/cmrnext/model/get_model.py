from argparse import Namespace

from cmrnext.model.RAFT.raft import RAFT_cmrnext


def get_model(_config):
    raft_args = Namespace()
    raft_args.small = False
    raft_args.iters = 12
    model = RAFT_cmrnext(raft_args,
                        use_reflectance=_config['use_reflectance'],
                        with_uncertainty=_config['uncertainty'],
                        fourier_levels=_config['fourier_levels'],
                        unc_type=_config['der_type'],
                        unc_freeze=_config["unc_freeze"],
                        context_encoder=_config["context_encoder"])
    return model
