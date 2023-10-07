import pandas as pd
import argparse
import os
import glob
import random
import numpy as np
import torch
from evaluate import eval_func
from multiprocessing import  Process

csv_name = "results_detail"
# for orginal version
def make_dis_dataset(data_name):
    if data_name == 'shapes3d':
        import data.ground_truth.shapes3d as dshapes3d
        return dshapes3d.Dataset(np.arange(0,480000))
    elif data_name == "mpi3d":
        import data.ground_truth.mpi3d as dmpi3d
        return dmpi3d.Dataset(np.arange(0,1036800))
    elif data_name == "cars3d":
        import data.ground_truth.cars3d as dcars3d
        return dcars3d.Dataset(np.arange(0,17568))
    else:
        raise NotImplementedError()

def mapping_func(folder_name):
    splits = folder_name.split("/")
    folder = os.path.join(*splits[:-1])
    file = splits[-1].replace(".json","")
    file = "epoch={:06}.npz".format(int(file) // 3750)
    return os.path.join(folder, file)


def evaluate_process(new_eval_path, eval_path):
    logdir = os.path.join(*new_eval_path.split("/")[:-2])
    
    results = {"folder": new_eval_path,"epoch":int(new_eval_path.split("=")[1].replace(".ckpt",""))}

    
    # num_samples = state['hyper_parameters']['batch_size'] * state['global_step']
    
    data_array = np.load(new_eval_path.replace("checkpoints/","dis_repre/").replace(".ckpt",".npz"))
    latents = torch.from_numpy(data_array["latents"])
    if not os.path.exists(os.path.join(logdir,"dis_metrics")):
        os.makedirs(os.path.join(logdir,"dis_metrics"))
    org_results = eval_func(label_dataset, latents, os.path.join(logdir, "dis_metrics"), int(data_array['num_samples']))
    results.update(org_results)
    # sub_df = pd.DataFrame(dict([(k,[v]) for k,v in results.items()]))
    # df = pd.concat([df,sub_df])
    # df.to_csv(f"{eval_path}/{csv_name}.csv")
    # os.remove(new_eval_path.replace("checkpoints/","dis_repre/").replace(".ckpt",".npz"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="",
        help="logger directory",
    )
    parser.add_argument(
        "-p",
        "--process_num",
        type=int,
        default=10,
        help="logger directory",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="shapes3d",
        help="dataset name",
    )
    conf = parser.parse_args()


    
    eval_path = conf.logdir
    # dis_eval_dataloader = model._train_dataloader(shuffle = False)
    label_dataset = make_dis_dataset(data_name=conf.dataset)
    
    while True:
        try:
            # if not os.path.exists(f"{eval_path}/{csv_name}.csv"):
            #     df = pd.DataFrame([])
            # else:
            #     df = pd.read_csv(f"{eval_path}/{csv_name}.csv")
            #     df = df.drop(['Unnamed: 0'], axis=1)
            #     print(df)
            mat_files = glob.glob(f"{eval_path}/*/*/*.json")

            npz_files = glob.glob(f"{eval_path}/*/*/*.npz")
            files = [npz_file.replace("dis_repre/","checkpoints/").replace(".npz",".ckpt") for npz_file in npz_files]
            # if len(df) == 0:
            #     record_files = []
            # else:
            #     record_files = list(df['folder'])

            if len(mat_files) == 0:
                record_files = []
            else:
                record_files = [mapping_func(mat.replace("dis_metrics/","checkpoints/")) for mat in mat_files]

            
            
            
            run_files = []
            cond = None
            for file in files:
                if file not in record_files:
                    run_files.append(file)
            
            if len(run_files) == 0:
                print("waiting ckpt...")
                pass
            else:
                random.shuffle(run_files)
                new_eval_path_list = run_files[:min(len(run_files), conf.process_num)]
                process_list = []
                for i,new_eval_path in enumerate(new_eval_path_list):
                    p = Process(target=evaluate_process,args=(new_eval_path,eval_path,)) #实例化进程对象
                    p.start()
                    process_list.append(p)

                for i in process_list:
                    p.join()

                print('Testig process end!!')
        except KeyboardInterrupt:
            raise NotImplementedError
        except:
            print("load failed, retrying!")
