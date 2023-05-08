#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from datetime import datetime, timedelta, timezone
import torch


def get_JST_time():
    JST = timezone(timedelta(hours=+9), "JST")
    dt_now = datetime.now(JST)
    dt_now = dt_now.strftime("%Y%m%d-%H%M%S")
    return dt_now


def model_device(model, device):
    print("device: ", device)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # make parallel
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model


def convert(seqs, kmer_dict, max_length):
    # Convert string list to numbers
    seq_idx = []
    if not max_length:
        max_length = max([len(i) for i in seqs])
    for s in seqs:
        # MASK for bases except AUTGC
        convered_seq = [kmer_dict[i] if i in kmer_dict.keys() else 1 for i in s] + [0] * (max_length - len(s))
        seq_idx.append(convered_seq)
    return seq_idx


def plot_result(
    y_true: np.array, y_est: np.array, lims=[-1.5, 20], mode="save", name=None
) -> None:
    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(5, 5))
    heatmap, xedges, yedges = np.histogram2d(
        y_true, y_est, bins=100, range=(lims, lims)
    )
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    cset = ax.imshow(
        heatmap.T, extent=extent, origin="lower", norm=LogNorm(), cmap="rainbow"
    )
    ax.plot(lims, lims, ls="--", color="black", alpha=0.5, label="y=x")
    ax.set_xlabel("target value", fontsize=15)
    ax.set_ylabel("prediction value", fontsize=15)
    ax.legend()

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(cset, cax=cax).ax.set_title("count")
    if mode == "show":
        plt.show()
    elif mode == "save":
        if name == None:
            dt_now = get_JST_time()
            fig.savefig(f"{dt_now}")
        else:
            fig.savefig(f"{name}")
        plt.close()


def remove_padding(target, output):
    pad = torch.where(target != -1)
    target = target[pad]
    output = output[pad]
    return target, output


def one_plot(target, output, length):
    if length > 440:
        plt.figure(figsize=(length // 100, 4))
    plt.plot(range(length), target[:length], label="target", color="b")
    plt.plot(range(length), output[:length], label="output", color="r")
    plt.legend()
    plt.xlabel("base position")
    plt.ylabel("accessibility")
    plt.show()


def num_to_base(seq):
    seq = seq.tolist()
    seq = list(map(str, seq))
    seq = [
        "".join(seq)
        .replace("0", "")
        .translate(str.maketrans({"1": "N", "2": "A", "3": "U", "4": "G", "5": "C"}))
    ]
    print(seq[0])
    print(seq[0][::-1])
