import os
import random
import logging
from enum import Enum
import fire
import numpy as np
from modules import reporter
import torch
from torch.utils import data as torch_data
from quantities import ms, second
from tqdm import tqdm
from h5py import File
from modules.topologies import SmallWorldTopology
from modules.pcritical import PCritical
from modules.utils import OneToNLayer
from ebdataset.audio import NTidigits
from modules.readout import (
    LinearWithBN,
    TimeBinningLayer,
    ExponentialFilterLayer,
    ReverseExponentialFilterLayer,
)
from modules.utils import unbatchifier, SpikeRecorder, StateRecorder

DATASET_PATH = os.environ["NTIDIGITS_DATASET_PATH"]
n_features = 64
n_classes = 11
dt = 1 * ms
_logger = logging.getLogger(__name__)


def rec_array_to_spike_train(sparse_spike_train):
    ts = sparse_spike_train.ts * second
    ts = (ts.rescale(dt.units) / dt).magnitude
    duration = np.ceil(np.max(ts)) + 1
    spike_train = torch.zeros((n_features, duration.astype(int)))
    spike_train[sparse_spike_train.addr, ts.astype(int)] = 1
    return spike_train


def collate_fn(samples):
    """Create a batch out of a list of tuple [(spike_train_tensor, str_label)]
    by zero-padding the spike trains"""
    max_duration = max([s[0].shape[-1] for s in samples])
    batch = torch.zeros(len(samples), n_features, max_duration)
    labels = []
    for i, s in enumerate(samples):
        batch[i, :, : s[0].shape[-1]] = s[0]
        labels.append(int(s[1].replace("z", "0").replace("o", "10")))
    return batch, torch.tensor(labels)


class ReadoutType(Enum):
    TIME_BINNING = "Time binning"
    EXPONENTIAL_FILTER = "Exponential filter"
    REVERSE_EXPONENTIAL_FILTER = "Reverse exponential filter"


def run_reservoir_ntidigits(
    nb_iters: int,
    device: torch.device,
    plasticity: bool = True,
    spectral_radius_norm: bool = False,
    readout_layer_type: ReadoutType = ReadoutType.TIME_BINNING,
    debug: bool = False,
    fix_weights_for_val: bool = True,
):
    _logger.info("Starting N-TIDIGITS classification experiment")

    reporter.log_tags(["N-TIDIGITS", "-".join(readout_layer_type.value.split(" "))])
    reporter.log_parameters(
        {
            "dt": dt,
            "plasticity": plasticity,
            "nb_iters": nb_iters,
        }
    )

    topology = SmallWorldTopology(
        reporter.log_parameters(
            SmallWorldTopology.Configuration(
                minicolumn_shape=(4, 4, 4),
                macrocolumn_shape=(2, 2, 2),
                minicolumn_spacing=1460,
                p_max=0.11,
                spectral_radius_norm=spectral_radius_norm,
                intracolumnar_sparseness=635,
                neuron_spacing=40,
                inhibitory_init_weight_range=(0.1, 0.3),
                excitatory_init_weight_range=(0.2, 0.5),
            )
        )
    )

    batch_size = 32
    reporter.log_parameter("batch_size", batch_size)
    data_loader_parameters = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": True,
        "timeout": 120,
        "collate_fn": collate_fn,
    }
    train_set = NTidigits(
        DATASET_PATH,
        is_train=True,
        transforms=rec_array_to_spike_train,
        only_single_digits=True,
    )
    val_set = NTidigits(
        DATASET_PATH,
        is_train=False,
        transforms=rec_array_to_spike_train,
        only_single_digits=True,
    )

    if debug:
        train_set = torch.utils.data.Subset(train_set, np.arange(50))
        val_set = torch.utils.data.Subset(val_set, np.arange(50))

    pcritical_configs: dict = reporter.log_parameters(
        {
            "alpha": 1e-2,
            "stochastic_alpha": False,
            "beta": 1e-5,
            "tau_v": 30 * ms,
            "tau_i": 5 * ms,
            "tau_v_pair": 5 * ms,
            "tau_i_pair": 0 * ms,
            "v_th": 1,
            "refractory_period": 2 * ms,
        }
    )

    n_neurons = topology.number_of_nodes()
    model = torch.nn.Sequential(
        OneToNLayer(N=3, dim_input=n_features, dim_output=n_neurons),
        PCritical(1, topology, dt=dt, **pcritical_configs),
    ).to(device)
    model[1].plasticity = plasticity

    if readout_layer_type == ReadoutType.TIME_BINNING:
        bin_size = 60  # ms
        reporter.log_parameter("Time bin size", bin_size * ms)
        convert_layer = TimeBinningLayer(bin_size, max_duration=2464, nb_of_neurons=n_neurons).to(device)
    elif readout_layer_type == ReadoutType.EXPONENTIAL_FILTER:
        exp_tau = 60
        reporter.log_parameter("Exp filter tau", exp_tau * dt)
        convert_layer = ExponentialFilterLayer(tau=exp_tau, nb_of_neurons=n_neurons).to(device)
    elif readout_layer_type == ReadoutType.REVERSE_EXPONENTIAL_FILTER:
        reverse_exp_tau = 60
        reporter.log_parameter("Reverse exp filter tau", reverse_exp_tau * dt)
        convert_layer = ReverseExponentialFilterLayer(tau=reverse_exp_tau, nb_of_neurons=n_neurons).to(device)

    if debug:
        nb_of_debug_steps = 5000
        spike_recorder = SpikeRecorder(
            "pcritical-tidigits-spike-recording.h5",
            model[0].W_t.t(),
            topology,
            nb_of_debug_steps,
        )
        weight_recorder = StateRecorder(
            "pcritical-tidigits-weight-recording.h5",
            nb_of_debug_steps,
            ("reservoir_weights", (n_neurons, n_neurons)),
        )
        debug_progress_bar = tqdm(total=nb_of_debug_steps)

    def input_and_reservoir_layers(x):
        """
        Compute post-reservoir state-space for input batch x
        NOTE: If x is a batch, plasticity will be merged during iterations
        For more accurate readings, process one sample at a time

        :param x: Input sample
        :return: x => input layer => reservoir layer => convert layer
        """
        x = x.to(device)
        current_batch_size = x.shape[0]  # 1 if unbatchifier active

        if not debug:
            model[1].batch_size = current_batch_size  # Will also reset neuron states (mem pot, cur)
        duration = x.shape[-1]
        convert_layer.reset()

        for t in range(duration):
            out_spikes = model(x[:, :, t])
            lsm_output = convert_layer(spikes=out_spikes, time=t, duration=duration)

            if debug:
                exit_early = not spike_recorder(x[:, :, t], out_spikes)
                exit_early &= not weight_recorder(model[1].W_rec)
                if exit_early:
                    exit(0)
                debug_progress_bar.update(1)

        return lsm_output

    reservoir_output_for_all_iters = []

    for iter_nb in range(nb_iters):
        reporter.log_metric("iteration", iter_nb)

        # -------- TRAINING PHASE --------
        train_generator = torch_data.DataLoader(train_set, shuffle=True, **data_loader_parameters)
        progress_bar = tqdm(train_generator, desc=f"train iter {iter_nb} / {nb_iters}", disable=debug)
        train_batches = [(unbatchifier(x, input_and_reservoir_layers), y) for (x, y) in progress_bar]

        # -------- VALIDATION PHASE --------
        val_gen = torch_data.DataLoader(val_set, shuffle=False, **data_loader_parameters)
        progress_bar = tqdm(val_gen, desc=f"val iter {iter_nb} / {nb_iters}", disable=debug)
        if fix_weights_for_val:
            model[1].plasticity = False
        val_batches = [(unbatchifier(x, input_and_reservoir_layers), y) for (x, y) in progress_bar]

        model[1].plasticity = plasticity

        reservoir_output_for_all_iters.append((train_batches, val_batches))

    return reservoir_output_for_all_iters, convert_layer.number_of_features()


def run_classification_ntidigits(
    reservoir_output, device: torch.device, n_features, weight_decay: float = 0.0, lr=1e-3, with_batch_norm=True
):
    train_accuracy_for_iters = []
    val_accuracy_for_iters = []
    if with_batch_norm:
        classifier = LinearWithBN(n_features, n_classes).to(device)
    else:
        classifier = torch.nn.Linear(n_features, n_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    reporter.log_parameters({"optimizer": "Adam", "weight_decay": weight_decay, "lr": lr})
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    pbar = tqdm(reservoir_output)
    for train_batches, val_batches in pbar:
        nb_accurate = 0
        nb_elems = 0
        sum_loss = 0
        for train_batch in train_batches:
            x, y = train_batch
            optimizer.zero_grad()
            net_out = classifier(x)
            preds = torch.argmax(net_out.detach(), dim=1).cpu()
            loss = loss_fn(net_out, y.to(device))
            loss.backward()
            optimizer.step()
            sum_loss += loss.cpu().detach().item()
            nb_accurate += torch.sum(preds == y).item()
            nb_elems += len(y)

        train_acc = nb_accurate / nb_elems
        pbar.set_postfix(train_loss=sum_loss / len(train_batches), train_acc=train_acc)
        reporter.log_metric("train_accuracy", train_acc)
        reporter.log_metric("train_loss", sum_loss / len(train_batches))
        train_accuracy_for_iters.append(train_acc)

        with torch.no_grad():
            nb_accurate = 0
            nb_elems = 0
            for val_batch in val_batches:
                x, y = val_batch
                net_out = classifier(x)
                preds = torch.argmax(net_out.detach(), dim=1).cpu()
                nb_accurate += torch.sum(preds == y).item()
                nb_elems += len(y)
            val_acc = nb_accurate / nb_elems
            pbar.set_postfix(val_acc=val_acc)
            reporter.log_metric("val_accuracy", val_acc)
            val_accuracy_for_iters.append(val_acc)

            if isinstance(classifier, torch.nn.Sequential) and isinstance(classifier[0], torch.nn.BatchNorm1d):
                # Reset batch-norm parameters so we do use them for training
                classifier[0].reset_running_stats()

    return train_accuracy_for_iters, val_accuracy_for_iters


def main(
    seed=0x1B,
    debug=False,
    plasticity=True,
    spectral_radius=False,
    nb_iters=10,
    weight_decay=0.0,
    fix_weights_for_val=False,
    save_reservoir_output=False,
):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Init logging
    reporter.init(
        "pcritical",
        backend=reporter.Backend.Logging,
        debug=debug,
    )
    reporter.log_parameter("seed", seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Using device type %s", str(device))

    # Run experiment
    reservoir_output, n_features = run_reservoir_ntidigits(
        nb_iters,
        plasticity=plasticity,
        spectral_radius_norm=spectral_radius,
        device=device,
        debug=debug,
        fix_weights_for_val=fix_weights_for_val,
    )

    if save_reservoir_output:
        import pickle

        pickle.dump((reservoir_output, n_features), open(f"ntidigits_reservoir_output_{seed}.pkl", "wb"))

    train_accuracies, test_accuracies = run_classification_ntidigits(
        reservoir_output, n_features=n_features, weight_decay=weight_decay, device=device
    )
    if not debug:
        reporter.dump_results(train_accuracies=train_accuracies, test_accuracies=test_accuracies)


if __name__ == "__main__":
    fire.Fire(main)
