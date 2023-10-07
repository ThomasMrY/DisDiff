from sklearn.decomposition import PCA
import numpy as np
import torch
import math
import gin
import os
import json

def write_text(result_dict,file):
    with open(file,'w+') as f:
        json.dump(result_dict,f)

def encode_data(data, model, latent_encoder):
    im_code = model.encode(data)
    old_shape = im_code.shape
    im_emb, _ = model.emb(im_code.detach())
    im_code_new = im_emb.view(im_code.shape[0],im_code.shape[1],-1).permute(0,2,1)

    my_latents = latent_encoder(im_code_new)
    return my_latents, old_shape

def enc_func(test_loader, model, perceiver_enc):
    latents_list = []
    from tqdm import tqdm
    with torch.no_grad():
        for x_curr, _ in tqdm(test_loader):
            x_curr = x_curr.cuda()
            my_latents, _ = encode_data(x_curr, model, perceiver_enc)
            latents_list.append(my_latents.detach().cpu())
            del x_curr, my_latents
    latents = torch.cat(latents_list, dim=0)
    return latents

def eval_func(eval_dataset, data_tensor, metric_folder, it, preflix=""):

    pca_rep = np.reshape(data_tensor, (data_tensor.shape[0],-1))
    MED_score = True
    beta_VAE_score = False
    dci_score = False
    factor_VAE_score = False
    MIG_score = False
    total_results_dict = {}
    def _representation(x):
        return pca_rep[x]
    if MED_score:
        with gin.unlock_config():
            from evaluation.metrics.med import compute_med
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("med.num_train", 10000)
            gin.bind_parameter("med.num_test", 10000)
            gin.bind_parameter("med.topk", 2)
            gin.bind_parameter("discretizer.discretizer_fn", _histogram_discretize)
            gin.bind_parameter("discretizer.num_bins", 20)
        result_dict = compute_med(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("MED score:" + str(result_dict))
        total_results_dict["MED" + preflix] = result_dict
    if beta_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter("beta_vae_sklearn.batch_size", 64)
            gin.bind_parameter("beta_vae_sklearn.num_train", 10000)
            gin.bind_parameter("beta_vae_sklearn.num_eval", 5000)
        result_dict = compute_beta_vae_sklearn(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("beta VAE score:" + str(result_dict))
        total_results_dict["beta_VAE" + preflix] = result_dict
    if dci_score:
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter("dci.num_train", 10000)
            gin.bind_parameter("dci.num_test", 5000)
        result_dict = compute_dci(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("dci score:" + str(result_dict))
        total_results_dict["dci" + preflix] = result_dict
    if MIG_score:
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("mig.num_train",10000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = compute_mig(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("MIG score:" + str(result_dict))
        total_results_dict["MIG" + preflix] = result_dict
    if factor_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter("factor_vae_score.num_variance_estimate",10000)
            gin.bind_parameter("factor_vae_score.num_train",10000)
            gin.bind_parameter("factor_vae_score.num_eval",5000)
            gin.bind_parameter("factor_vae_score.batch_size",64)
            gin.bind_parameter("prune_dims.threshold",0.05)
        result_dict = compute_factor_vae(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("factor VAE score:" + str(result_dict))
        total_results_dict["factor_VAE" + preflix] = result_dict
    write_text(total_results_dict,metric_folder + f"/{it}.json")
    return total_results_dict

def eval_func_test(eval_dataset, data_list, metric_folder, it):
    pca_list = []
    for i in range(data_list[0].shape[1]):
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(torch.cat([data_tensor[:,i,:] for data_tensor in data_list], dim=0).numpy())
        pca_list.append(pca_result)
    pca_rep = np.concatenate(pca_list, axis=1)

    beta_VAE_score = True
    dci_score = True
    factor_VAE_score = True
    MIG_score = True
    total_results_dict = {}
    def _representation(x):
        return pca_rep[x]
    if beta_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter("beta_vae_sklearn.batch_size", 64)
            gin.bind_parameter("beta_vae_sklearn.num_train", 10000)
            gin.bind_parameter("beta_vae_sklearn.num_eval", 5000)
        result_dict = compute_beta_vae_sklearn(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("beta VAE score:" + str(result_dict))
        total_results_dict["beta_VAE_score"] = result_dict
    if dci_score:
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter("dci.num_train", 10000)
            gin.bind_parameter("dci.num_test", 5000)
        result_dict = compute_dci(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("dci score:" + str(result_dict))
        total_results_dict["dci_score"] = result_dict
    if MIG_score:
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("mig.num_train",10000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = compute_mig(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("MIG score:" + str(result_dict))
        total_results_dict["MIG_score"] = result_dict
    if factor_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter("factor_vae_score.num_variance_estimate",10000)
            gin.bind_parameter("factor_vae_score.num_train",10000)
            gin.bind_parameter("factor_vae_score.num_eval",5000)
            gin.bind_parameter("factor_vae_score.batch_size",64)
            gin.bind_parameter("prune_dims.threshold",0.05)
        result_dict = compute_factor_vae(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("factor VAE score:" + str(result_dict))
        total_results_dict["factor_VAE_score"] = result_dict
        write_text(total_results_dict,metric_folder + f"/{it}.json")