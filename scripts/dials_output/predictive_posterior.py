import argparse
from pathlib import Path

import h5py
import torch

from refltorch.io import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to get the predictive posterior for trained model"
    )
    parser.add_argument("--run-dir", type=str, help="help text")
    return parser.parse_args()


def main():
    args = parse_args()

    # loading config file
    run_dir = Path(args.run_dir)
    run_metadata = list(run_dir.glob("run_metadata.yaml"))[0]
    config = load_config(run_metadata)

    # w&b log-dir
    wandb_log = Path(config["wandb"]["log_dir"]).parent

    # getting h5 files for each epoch

    h5_all_epochs = list((wandb_log / "predictions").glob("**/*.h5"))

    # Getting torch distributions

    # TEMP
    # NOTE: Using `h5_file` as placeholder

    h5_file = h5_all_epochs[0]

    f = h5py.File(h5_file, "r")

    # number of shoebox distributions to use
    n = 100

    # number of monte carlo samples
    n_samples = 100

    log_ppds = []
    for h in h5_all_epochs:
        f = h5py.File(h, "r")

        # qbg
        qbg_concentration = f["qbg_params.concentration"][:n]
        qbg_concentration = torch.from_numpy(qbg_concentration)
        qbg_rate = f["qbg_params.rate"][:n]
        qbg_rate = torch.from_numpy(qbg_rate)
        qbg = torch.distributions.Gamma(qbg_concentration, qbg_rate)

        # qi
        qi_concentration = f["qi_params.concentration"][:n]
        qi_concentration = torch.from_numpy(qi_concentration)
        qi_rate = f["qi_params.rate"][:n]
        qi_rate = torch.from_numpy(qi_rate)
        qi = torch.distributions.Gamma(qi_concentration, qi_rate)

        # qp
        qp_concentration = f["concentration"][:n]
        qp_concentration = torch.from_numpy(qp_concentration)
        qp = torch.distributions.Dirichlet(qp_concentration)

        # getting rates
        zi = qi.sample([n_samples]).unsqueeze(-1).permute(1, 0, 2)
        zbg = qbg.sample([n_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.sample([n_samples]).permute(1, 0, 2)
        rates = zi * zp + zbg

        # Now we parameterize a Poisson distribution
        poisson = torch.distributions.Poisson(rates)

        # observed data
        y = torch.from_numpy(f["counts"][:n]).unsqueeze(1)

        # Log likelihood per sample, per pixel
        log_p_y_given_lambda = poisson.log_prob(y)  # (N, S, P)

        # Sum over pixels
        log_p_y_given_lambda = log_p_y_given_lambda.sum(dim=-1)  # (N, S)

        # Log posterior predictive via log-mean-exp
        log_ppd = torch.logsumexp(log_p_y_given_lambda, dim=1) - torch.log(
            torch.tensor(n_samples, dtype=log_p_y_given_lambda.dtype)
        )  # (N,)
        log_ppds.append(log_ppd)

    mean_epochs = torch.stack(log_ppds).mean(dim=1)
    mean_epochs = torch.stack(log_ppds).std(dim=1)


def _get_h5_files(wandb_log):
    pass


if __name__ == "__main__":
    main()
