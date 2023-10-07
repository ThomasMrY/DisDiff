import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
from ldm.util import get_obj_from_str,instantiate_from_config
import glob
import random
from tqdm import tqdm
from einops import rearrange
import pandas as pd
from utils import load_yaml, move_to_cuda, calculate_ssim, calculate_lpips, calculate_mse, set_worker_seed_builder, init_process
from torchvision.utils import make_grid, save_image
import json

csv_name = "results_detail"
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def mapping_func(folder_name):
    splits = folder_name.split("/")
    folder = os.path.join(*splits[:-1])
    file = splits[-1].replace(".json","")
    file = "epoch={:06}.npz".format(int(file) // epoch_steps)
    return os.path.join(folder, file)

import lpips

if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    with open(".gitignore", "r") as f:
        lines = f.readlines()
        if opt.logdir + "\n" not in lines:
            lines = [opt.logdir + "\n"] + lines
    with open(".gitignore", "w") as f:
        f.writelines(lines)
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    # try:
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    config = OmegaConf.merge(*configs)
    
    # data
    config.data.params.pop("train")
    config.data.params.test["params"] = {}
    config.data.params.test["params"]['test'] = True
    # config.data.params["test"] = True
    dataset = instantiate_from_config(config.data)
    dataset.prepare_data()
    dataset.setup()
    dis_eval_dataloader = dataset.test_dataloader()
    model = instantiate_from_config(config.model)

    if "shapes3d" in opt.logdir:
        epoch_steps = 480000//128
    elif "mpi3d" in opt.logdir:
        epoch_steps = 1036800//128
    elif "cars3d" in opt.logdir:
        epoch_steps = 175680//128 + 1
    while True:
        # try:
            # if not os.path.exists(f"{opt.logdir}/{csv_name}.csv"):
            #     df = pd.DataFrame([])
            # else:
            #     df = pd.read_csv(f"{opt.logdir}/{csv_name}.csv")
            #     df = df.drop(['Unnamed: 0'], axis=1)
            #     # print(df)
        run_files = []
        for folder in os.listdir(opt.logdir):
            # if "CelebA" in folder:
            file = sorted(os.listdir(os.path.join(opt.logdir, folder,"checkpoints")))[-2]
            run_files.append(os.path.join(opt.logdir, folder, "checkpoints", file))
                
                
        
        
        if len(run_files) == 0:
            pass
            # with torch.no_grad():
            #     for i in range(1000):
            #         x_T = model.x_T[:size].cuda()
            #         sample_x0 = sample_x0_all[:size].cuda()
            #         gen = model.eval_sampler.sample(model=model.model,
            #                                             noise=x_T,
            #                                             cond=cond,
            #                                             x_start=sample_x0)
            #     del x_T, sample_x0, gen
        else:
            selected = random.randint(0, len(run_files)-1)
            new_eval_path = run_files[selected]
            run_files.pop(selected)
            file = new_eval_path.split("/")[-1]
            os.makedirs(new_eval_path.replace(f"checkpoints/{file}","tad_metrics"),exist_ok=True)
            sim_logdir = os.path.join(*new_eval_path.split("/")[:-2])
            model.init_from_ckpt(new_eval_path,only_model=False)
            
            model.cuda()


            seed = 30 # FIXED AT 30 FOR ALL EXPERIMENTS
            import random
            random.seed(seed)
            import numpy as np
            import matplotlib.pyplot as plt
            from ae_utils_exp import multi_t, LatentClass, aurocs_search, tags
            from torchvision.transforms import Compose

            np.random.seed(seed)
            torch.manual_seed(seed)
            from torchvision.datasets import CelebA
            import torchvision.transforms as tforms

            tform = tforms.Compose([tforms.Resize(96), tforms.CenterCrop(64), tforms.ToTensor()])
            # tform = tforms.Compose([tforms.Resize(96), tforms.CenterCrop(64), tforms.ToTensor(), tforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # tform = tforms.Compose([
            #     tforms.Resize(64),
            #     tforms.CenterCrop(64),
            #     tforms.ToTensor(),
            #     tforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])

            eval_bs = 1000
            # set up dataset for eval
            # dataset_eval = CelebA(root='../../../guided-diffusion/datasets/CelebA', split='all', target_type='attr', download=False, transform=tform)

            # dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=eval_bs, shuffle=True, drop_last=False,num_workers=8)
            file = np.load("test_celeba.npz")
            data = torch.from_numpy(file['data'])
            targ = torch.from_numpy(file['targ'])
            au_result, base_rates_raw, targ = aurocs_search(data, targ, model)
            base_rates = base_rates_raw.where(base_rates_raw <= 0.5, 1. - base_rates_raw)

            # fig, ax = plt.subplots(8, 5, figsize=(16, 16))
            # print the ind, tag, max auroc, arg max auroc, norm_diff
            max_aur, argmax_aur = torch.max(au_result.clone(), dim=1)
            norm_diffs = torch.zeros(40).cuda()
            aurs_diffs = torch.zeros(40).cuda()
            for ind, tag, max_a, argmax_a, aurs in zip(range(40), tags, max_aur.clone(), argmax_aur.clone(), au_result.clone()):
                norm_aurs = (aurs.clone() - 0.5) / (aurs.clone()[argmax_a] - 0.5)
                aurs_next = aurs.clone()
                aurs_next[argmax_a] = 0.0
                aurs_diff = max_a - aurs_next.max()
                aurs_diffs[ind] = aurs_diff
                norm_aurs[argmax_a] = 0.0
                norm_diff = 1. - norm_aurs.max()
                norm_diffs[ind] = norm_diff
                print("{}\t\t Lat: {}\t Max: {:1.3f}\t ND: {:1.3f}".format(tag, argmax_a.item(), max_a.item(), norm_diff.item()))
                plt_ind = ind//5, ind%5
                # ax[plt_ind].set_ylim((0.5, max_a.item() + 0.05))
                # ax[plt_ind].set_title(tag)
                # ax[plt_ind].set_ylabel("AUROC")
                # ax[plt_ind].set_xlabel("Latent Variable")
                # ax[plt_ind].bar(range(aurs.shape[0]), aurs)
                # ax[plt_ind].grid(which='both', axis='y')
                assert aurs.max() == max_a



            # calculate mutual information shared between attributes
            # determine which share a lot of information with each other
            with torch.no_grad():
                not_targ = 1 - targ
                j_prob = lambda x, y: torch.logical_and(x, y).sum() / x.numel()
                mi = lambda jp, px, py: 0. if jp == 0. or px == 0. or py == 0. else jp*torch.log(jp/(px*py))

                # Compute the Mutual Information (MI) between the labels
                mi_mat = torch.zeros((40, 40)).cuda()
                for i in range(40):
                    # get the marginal of i
                    i_mp = targ[:, i].sum() / targ.shape[0]
                    for j in range(40):
                        j_mp = targ[:, j].sum() / targ.shape[0]
                        # get the joint probabilities of FF, FT, TF, TT
                        # FF
                        jp = j_prob(not_targ[:, i], not_targ[:, j])
                        pi = 1. - i_mp
                        pj = 1. - j_mp
                        mi_mat[i][j] += mi(jp, pi, pj)
                        # FT
                        jp = j_prob(not_targ[:, i], targ[:, j])
                        pi = 1. - i_mp
                        pj = j_mp
                        mi_mat[i][j] += mi(jp, pi, pj)
                        # TF
                        jp = j_prob(targ[:, i], not_targ[:, j])
                        pi = i_mp
                        pj = 1. - j_mp
                        mi_mat[i][j] += mi(jp, pi, pj)
                        # TT
                        jp = j_prob(targ[:, i], targ[:, j])
                        pi = i_mp
                        pj = j_mp
                        mi_mat[i][j] += mi(jp, pi, pj)

                # fig, ax = plt.subplots(1, 2)
                # im = ax[0].imshow(mi_mat)
                # fig.colorbar(im, ax=ax[0], shrink=0.6)
                # mi_mat_ent_norm = mi_mat/mi_mat.diag().unsqueeze(1)
                # im = ax[1].imshow(mi_mat_ent_norm)
                # fig.colorbar(im, ax=ax[1], shrink=0.6)
                
                # plt.figure(figsize=(10, 7))
                # mi_comp = (mi_mat.sum(dim=1) - mi_mat.diag())/mi_mat.diag()
                # plt.bar(range(len(tags)), mi_comp, tick_label=tags)
                # plt.xticks(rotation=90)
                # plt.title("Total Mutual Information")
                
                # plt.figure(figsize=(10, 7))
                mi_maxes, mi_inds = (mi_mat * (1 - torch.eye(40).cuda())).max(dim=1)
                ent_red_prop = 1. - (mi_mat.diag() - mi_maxes) / mi_mat.diag()
                # plt.bar(range(len(tags)), ent_red_prop, tick_label=tags)
                # plt.xticks(rotation=90)
                # plt.title("Proportion of Entropy Reduced by Another Trait")
                # plt.grid(axis='y')
                print(mi_mat.diag())

            thresh = 0.75
            ent_red_thresh = 0.2

            # calculate Average Norm AUROC Diff when best detector score is at a certain threshold
            filt = (max_aur >= thresh).logical_and(ent_red_prop <= ent_red_thresh)
            # calculate Average Norm AUROC Diff when best detector score is at a certain threshold
            aurs_diffs_filt = aurs_diffs[filt]
            print(len(aurs_diffs_filt))

            # plt.figure(figsize=(10, 7))
            # plt.ylim((0.0, 1.0))
            # plt.title("Total AUROC Diff: {:1.3f} at Thresh: {:1.2f}".format(aurs_diffs_filt.sum(), thresh))
            # plt.ylabel("AUROC Difference")
            # plt.xlabel("Attribute")
            # plt.xticks(rotation=90)
            # plt.bar(range(aurs_diffs.shape[0]), aurs_diffs, tick_label=tags)
            # plt.grid(which='both', axis='y')
            
            print("TAD SCORE: ", aurs_diffs_filt.sum().item(), "Attributes Captured: ", len(aurs_diffs_filt))
            save_file = new_eval_path.replace(f"checkpoints","tad_metrics").replace("ckpt","json")
            with open(save_file,"w+") as f:
                json.dump({"TAD SCORE: ":aurs_diffs_filt.sum().item(), "Attributes Captured: ":len(aurs_diffs_filt)},f)
                
                
            
