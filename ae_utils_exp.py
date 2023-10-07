import torch
import math
from tqdm import tqdm
import time
class Normalize(object):
    def __init__(self, mean, std, ndim=2):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        for d in range(ndim):
            self.mean = self.mean.unsqueeze(-1)
            self.std = self.std.unsqueeze(-1)

    def __call__(self, tensor):
        return tensor.sub(self.mean.to(tensor.device)).div(self.std.to(tensor.device))
    
class UnNormalize(object):
    def __init__(self, norm):
        super(UnNormalize, self).__init__()
        self.mean = norm.mean
        self.std = norm.std

    def __call__(self, tensor):
        return self.scale_inorm(tensor).add(self.mean.to(tensor.device))

    def scale_inorm(self, tensor):
        return tensor.mul(self.std.to(tensor.device))
    
beam_s2s2_norm = Normalize((0.318,), (0.4168), ndim=1)
beam_s2s2_inorm = UnNormalize(beam_s2s2_norm)
    
celeba_norm = Normalize((0.5337, 0.4157, 0.3562), (0.2956, 0.2581, 0.2477))
celeba_inorm = UnNormalize(celeba_norm)

dsprites_norm = Normalize((0.0429,), (0.2026,))
dsprites_inorm = UnNormalize(dsprites_norm)

tags = ["5_o_Clock_Shadow",
"Arched_Eyebrows",
"Attractive",
"Bags_Under_Eyes",
"Bald",
"Bangs",
"Big_Lips",
"Big_Nose",
"Black_Hair",
"Blond_Hair",
"Blurry",
"Brown_Hair",
"Bushy_Eyebrows",
"Chubby",
"Double_Chin",
"Eyeglasses",
"Goatee",
"Gray_Hair",
"Heavy_Makeup",
"High_Cheekbones",
"Male",
"Mouth_Slightly_Open",
"Mustache",
"Narrow_Eyes",
"No_Beard",
"Oval_Face",
"Pale_Skin",
"Pointy_Nose",
"Receding_Hairline",
"Rosy_Cheeks",
"Sideburns",
"Smiling",
"Straight_Hair",
"Wavy_Hair",
"Wearing_Earrings",
"Wearing_Hat",
"Wearing_Lipstick",
"Wearing_Necklace",
"Wearing_Necktie",
"Young"]

def multi_t(a, _f, _t):
    assert _f >= 0
    assert _t >= 0
    assert _f < a.dim()
    assert _t < a.dim()
    while(_f != _t):
        if _f < _t:
            a = a.transpose(_f, _f+1)
            _f += 1
        else:
            a = a.transpose(_f-1, _f)
            _f -= 1
    return a

def covariance(X, Y): # where each is shape (N, L)
    cov = (X*Y).mean(dim=0) - X.mean(dim=0)*Y.mean(dim=0)
    return cov

def one_cold(dim, ind):
    out = torch.ones(dim)
    out[ind] = 0
    return out

def s_init(module):
    torch.nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
    return module

def kai_norm(module, nonlinearity='relu'):
    torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity=nonlinearity)
    return module

class InpNorm1D(torch.nn.BatchNorm2d):
    
    def __init__(self, affine=False):
        super(InpNorm1D, self).__init__(1, affine=affine)
    
    def __call__(self, x):
        return self.forward(x.view(x.shape[0], 1, 1, -1)).squeeze()
    
class MinMaxNormalize(object):
    def __init__(self, _min=0, _max=1):
        super(MinMaxNormalize, self).__init__()
        self._min = _min
        self._max = _max
        
    def __call__(self, tensor):
        adj = tensor - tensor.min().item()
        maxval = adj.max().item() # non-negative
        if maxval != 0.0:
            adj = adj.div(maxval) # normalize to [0.0, 1.0]
        return adj
    
class InvNorm(object):
    
    def __init__(self, norm):
        super(InvNorm, self).__init__()
        self.mean = 0.
        self.std = 1.
        if hasattr(norm, 'running_mean'):
            self.mean = norm.running_mean
        if hasattr(norm, 'running_var'):
            self.std = norm.running_var.sqrt()
        
    def __call__(self, x):
        # expect batch, ...
        dim = x.dim()
        mean = self.mean.clone()
        std = self.std.clone()
        for i in range(dim - 2):
            mean.unsqueeze_(-1)
            std.unsqueeze_(-1)
        return (x + mean) * std
    

# function that takes a lists of latent indices, thresholds, and signs for classification
class LatentClass(object):
    
    def __init__(self, targ_ind, lat_ind, is_pos, thresh, __max, __min):
        super(LatentClass, self).__init__()
        self.targ_ind = targ_ind
        self.lat_ind = lat_ind
        self.is_pos = is_pos
        self.thresh = thresh
        self._max = __max
        self._min = __min
        self.it = list(zip(self.targ_ind, self.lat_ind, self.is_pos, self.thresh))
        
    def __call__(self, z):
        # expect z to be [batch, z_dim]
        out = torch.ones((z.shape[0], 40)).cuda()
        for t_i, l_i, is_pos, t in self.it:
            ma, mi = self._max[l_i], self._min[l_i]
            thr = t * (ma - mi) + mi
            res = (z[:, l_i] >= thr if is_pos else z[:, l_i] < thr).type(torch.int)
            out[:, t_i] = res
        return out
    
def calculate_auroc(targ, targ_ind, lat_ind, z, _ma, _mi, stepsize=0.1):
    thr = torch.arange(0.0, 1.0001, step=stepsize)
    total = targ.shape[0]
    pos_total = targ.sum(dim=0)[targ_ind]
    neg_total = total - pos_total
    p_fpr_tpr = torch.zeros((thr.shape[0], 2)).cuda()
    n_fpr_tpr = torch.zeros((thr.shape[0], 2)).cuda()
    for i, t in enumerate(thr):
        local_lc = LatentClass([targ_ind], [lat_ind], [True], [t], _ma, _mi)
        pred = local_lc(z.clone()).to(targ.device)
        p_tp = torch.logical_and(pred == targ, pred).sum(dim=0)[targ_ind]
        p_fp = torch.logical_and(pred != targ, pred).sum(dim=0)[targ_ind]
        p_fpr_tpr[i][0] = p_fp/neg_total
        p_fpr_tpr[i][1] = p_tp/pos_total
        local_lc = LatentClass([targ_ind], [lat_ind], [False], [t], _ma, _mi)
        pred = local_lc(z.clone()).to(targ.device)
        n_tp = torch.logical_and(pred == targ, pred).sum(dim=0)[targ_ind]
        n_fp = torch.logical_and(pred != targ, pred).sum(dim=0)[targ_ind]
        n_fpr_tpr[i][0] = n_fp/neg_total
        n_fpr_tpr[i][1] = n_tp/pos_total
    # p_fpr_tpr = (p_fpr_tpr.cpu().sort(dim=0)[0]).cuda()
    # n_fpr_tpr = (n_fpr_tpr.cpu().sort(dim=0)[0]).cuda()
    p_fpr_tpr = p_fpr_tpr.sort(dim=0)[0]
    n_fpr_tpr = n_fpr_tpr.sort(dim=0)[0]
    p_dists = p_fpr_tpr[1:, 0] - p_fpr_tpr[:-1, 0]
    p_area = (p_fpr_tpr[1:, 1] * p_dists).sum()
    n_dists = n_fpr_tpr[1:, 0] - n_fpr_tpr[:-1, 0]
    n_area = (n_fpr_tpr[1:, 1] * n_dists).sum()
    return p_area, n_area

def aurocs(_z, targ, targ_ind, _ma, _mi):
    # perform a grid search of lat_ind to find the best classification metric
    aurocs = torch.ones(_z.shape[1]).cuda() * 0.5 # initialize as random guess
    for lat_ind in tqdm(range(_z.shape[1])):
        if _ma[lat_ind] - _mi[lat_ind] > 0.2:
            p_auroc, n_auroc = calculate_auroc(targ, targ_ind, lat_ind, _z.clone(), _ma, _mi)
            m_auroc = max(p_auroc, n_auroc)
            aurocs[lat_ind] = m_auroc
            #print("{}\t{:1.3f}".format(lat_ind, m_auroc))
    return aurocs

def aurocs_search(data, targ, ae):
    aurocs_all = torch.ones((40, ae.cond_stage_model.latent_dim)).cuda() * 0.5
    with torch.no_grad():
        # data, targ = next(iter(dl))
        data, targ = data.to(ae.device), targ.to(ae.device)
        base_rates_all = targ.sum(dim=0)
        base_rates_all = base_rates_all / targ.shape[0]
        out = ae.cond_stage_model(data)
        _ma = out.max(dim=0)[0]
        _mi = out.min(dim=0)[0]
        if type(ae) is VAE:
            _ma = ae.mu.max(dim=0)[0]
            _mi = ae.mu.min(dim=0)[0]
        for i in range(40):
            print(i)
            start = time.time()
            aurocs_all[i] = aurocs(out.clone(), targ, i, _ma, _mi)
            print(time.time() - start)
    return aurocs_all.cuda(), base_rates_all.cuda(), targ.cuda()

          
class DisentanglementMetric(torch.nn.Module):
    
    def __init__(self, n_latent, n_data_fact, device='cpu', lr=0.01):
        super(DisentanglementMetric, self).__init__()
        self.w = kai_norm(torch.nn.Linear(n_latent, n_data_fact), nonlinearity='linear')
        self.optim = torch.optim.Adagrad(self.w.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.to(device)
        
    def forward(self, x):
        self.out = self.w(x)
        return torch.max(self.out, dim=1)[1] # return latent index predictions
    
    def set_lr(self, lr):
        self.optim = torch.optim.Adagrad(self.w.parameters(), lr=lr)
    
    def step(self, y):
        self.w.zero_grad()
        self.criterion.zero_grad()
        loss = self.criterion(self.out, y)
        loss.backward()
        self.optim.step()
        return loss.item()
    
    def fit_batch(self, true_factor, lat_batch):
        label = torch.full((1,), true_factor, dtype=torch.long)
        prediction = self.forward(lat_batch.mean(dim=0).unsqueeze(0))
        loss = self.step(label)
        return loss
        
        
class Predictor(torch.nn.Module):
    
    def __init__(self, config=(10, 20, 20)):
        super(Predictor, self).__init__()
        self.ops = []
        for i in range(len(config)-1):
            self.ops.append(s_init(torch.nn.Linear(config[i], config[i+1], bias=True)))
            self.ops.append(torch.nn.SELU(inplace=True))
        self.ops.append(s_init(torch.nn.Linear(config[-1], 1, bias=True)))
        self.op = torch.nn.Sequential(*self.ops)
        
    def forward(self, x):
        return self.op(x)
    
class PredictorEnsemble(torch.nn.Module):
    
    def __init__(self, n_preds=2, p_h_config=(40, 40)):
        super(PredictorEnsemble, self).__init__()
        self.n_preds = n_preds
        self.preds = torch.nn.ModuleList([Predictor(config=(n_preds,) + p_h_config) for i in range(n_preds)])
        
    def forward(self, latent_var):
        predictions = torch.empty_like(latent_var)
        for i in range(self.n_preds): # mask the ground truth
            mask = one_cold(latent_var.shape, (..., i)).to(latent_var.device)
            preds_out = self.preds[i](latent_var * mask).squeeze()
            predictions[..., i] = preds_out
        return predictions

    
class AutoEncoder(torch.nn.Module):
    
    def __init__(self, inp_norm, enc, dec, device, \
                 z_dim=2, p_h_config=(40, 40), z_act=torch.nn.Sigmoid(), inp_inorm=None):
        super(AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.enc = enc
        self.dec = dec
        self.device = device
        self.z_act = z_act
        self.inp_norm = inp_norm
        self.inp_inorm = inp_inorm
        self.preds = PredictorEnsemble(n_preds=self.z_dim, p_h_config=p_h_config)
        self.ae_optim=None
        self.preds_optim=None
        self.to(device)
        
    def create_encoding(self, x):
        self.x_norm = self.inp_norm(x)
        return self.z_act(self.enc(self.x_norm))
        
    def forward(self, x):
        self.z = self.create_encoding(x)
        self.z_pred = self.preds(self.z)
        self.x_pred = self.dec(self.z)
        return self.x_pred
    
    def compute_loss_ae(self, expected):
        # compute reconstruction loss
        rec_loss = 0.5*self.mse(self.x_pred, expected.detach())
        # compute covariance score
        adv_loss = covariance(self.z.detach(), self.z_pred).sum()
        
        return rec_loss, adv_loss
    
    def compute_loss_preds(self):
        return 0.5*self.mse(self.z_pred, self.z.detach())
    
    def step_ae(self, expected, ar=0.0):
        self.enc.zero_grad()
        self.dec.zero_grad()
        rec_loss, adv_loss = self.compute_loss_ae(expected)
        loss = (1. - ar)*(rec_loss) + ar*adv_loss
        loss.backward()
        self.ae_optim.step()
        return rec_loss, adv_loss
    
    def step_preds(self):
        self.preds.zero_grad()
        preds_loss = self.compute_loss_preds()
        preds_loss.backward()
        self.preds_optim.step()
        return preds_loss
    
    def init_optim_objects(self, lr, pred_lr):
        self.ae_optim = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        self.preds_optim = torch.optim.Adam(self.preds.parameters(), lr=pred_lr)
    
    def fit(self, dataset, n_group, batch_per_group=10, lr=0.001, pred_lr=0.01, ar=0.0, \
              batch_size=100, preds_train_iters=5, generator_ae=None):
        self.train()
        # create loss objects and optimizers
        self.mse = torch.nn.MSELoss()
        if self.ae_optim is None or self.preds_optim is None:
            self.init_optim_objects(lr, pred_lr)
        # set up loss storage
        rec_loss = torch.zeros(n_group)
        adv_loss = torch.zeros(n_group)
        pred_loss = torch.zeros(n_group)
        # define samplers for the AE
        n_samples = batch_size*batch_per_group*n_group
        random_sampler_ae = torch.utils.data.RandomSampler(dataset, \
                      replacement=True, num_samples=n_samples, generator=generator_ae)
        batch_sampler_ae = torch.utils.data.BatchSampler(random_sampler_ae, batch_size=batch_size, drop_last=False)
        dataloader_ae = iter(torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_ae))
        
        for g in range(n_group):
            rec_loss_agg = 0.
            adv_loss_agg = 0.
            pred_loss_agg = 0.
            for b in range(batch_per_group):
                print("\rGroup: {}\t{:2.0f}%".format(g, 100*(b+1)/batch_per_group), end="")
                data, label = next(dataloader_ae)
                # push examples through the autoencoder, get latent space activations
                ex = data.to(self.device)
                ex.requires_grad=False
                out = self.forward(ex)
                for p in range(preds_train_iters):
                    self.z_pred = self.preds(self.z.detach())
                    pred_loss_agg += self.step_preds() / preds_train_iters
                out = self.forward(ex)
                rec_loss_b, adv_loss_b = self.step_ae(self.x_norm.detach(), ar)
                rec_loss_agg += rec_loss_b
                adv_loss_agg += adv_loss_b
            rec_loss[g] = rec_loss_agg / batch_per_group
            adv_loss[g] = adv_loss_agg / batch_per_group
            pred_loss[g] = pred_loss_agg / batch_per_group
            print("\tRec: {:1.4f}\tAdv: {:1.4f}\tPred: {:1.4f}".format(\
                rec_loss[g], adv_loss[g], pred_loss[g]))
        return rec_loss.detach(), adv_loss.detach(), pred_loss.detach()
    
    def record_latent_space(self, dataset, batch_size=100, n_batches=5):
        n_elems = batch_size*n_batches
        z_scores = torch.empty((n_elems, self.z_dim))
        # sequential sampler
        dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False))
        i = 0
        self.eval()
        inp = None
        rec = None
        with torch.no_grad():
            while i < n_elems:
                data, label = dataloader.next()
                ex = data.to(self.device)
                if i + ex.shape[0] >= n_elems:
                    ex = ex[:n_elems-i]
                out = self.forward(ex)
                z_scores[i:i+ex.shape[0]] = self.z.detach()
                i += ex.shape[0]
                if inp is None:
                    inp = ex.detach()
                    rec = out.detach()
        invnorm = self.inp_inorm
        if invnorm is None:
            invnorm = InvNorm(self.inp_norm)
        rec = invnorm(rec)
        return z_scores, inp, rec
    
    
        
def compute_log_prob(sample, dstr_mu=None, dstr_sig=None):
    if dstr_mu is None:
        dstr_mu=torch.zeros_like(sample)
    if dstr_sig is None:
        dstr_sig=torch.ones_like(sample)
    dstr = torch.distributions.Normal(dstr_mu, dstr_sig)
    lprob = dstr.log_prob(sample)
    return lprob

def compute_kl_div(z, mu, sig):
    log_prob_enc = compute_log_prob(z, mu, sig)
    log_prob_prior = compute_log_prob(z)
    return log_prob_enc - log_prob_prior

def normal_dist(out):
    return torch.distributions.Normal(out, torch.ones_like(out))

def bernoulli_dist(out):
    return torch.distributions.Bernoulli(logits=out)

class VAE(torch.nn.Module):
    
    def __init__(self, inp_norm, enc, dec, device, z_dim, inp_inorm, rec_dstr='gaussian'):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec
        self.device = device
        self.z_dim = z_dim
        self.inp_norm = inp_norm
        self.inp_inorm = inp_inorm
        self.ae_optim = None
        if rec_dstr == 'gaussian':
            self.rec_dstr_func = normal_dist # was a lambda, changed it to function for pickling
        elif rec_dstr == 'bernoulli':
            self.rec_dstr_func = bernoulli_dist
        else:
            raise AttributeError("expected gaussian or bernoulli, but got {}".format(rec_dstr))
        self.to(device)
        
    def init_optim_objects(self, lr):
        self.ae_optim = torch.optim.Adagrad(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr)
        
    def forward(self, x):
        self.x_norm = self.inp_norm(x)
        self.mu, self.log_var = self.enc(self.x_norm) ### was log sigma
        z_dstr = torch.distributions.Normal(self.mu, torch.exp(0.5*self.log_var))
        self.z = z_dstr.rsample()
        self.x_pred = self.dec(self.z)
        return self.x_pred
    
    
    def fit(self, dataset, n_group, batch_per_group=10, lr=0.001, beta=1.0,\
              batch_size=100, generator_ae=None):
        self.train()
        # create loss objects and optimizers
        if self.ae_optim is None:
            self.init_optim_objects(lr)
        # set up loss storage
        rec_loss = torch.zeros(n_group)
        kl_loss = torch.zeros(n_group)
        # define samplers for the AE
        n_samples = batch_size*batch_per_group*n_group
        random_sampler_ae = torch.utils.data.RandomSampler(dataset, \
                      replacement=True, num_samples=n_samples, generator=generator_ae)
        batch_sampler_ae = torch.utils.data.BatchSampler(random_sampler_ae, batch_size=batch_size, drop_last=False)
        dataloader_ae = iter(torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_ae))
        
        for g in range(n_group):
            rec_loss_agg = 0.
            kl_loss_agg = 0.
            for b in range(batch_per_group):
                print("\rGroup: {}\t{:2.0f}%".format(g, 100*(b+1)/batch_per_group), end="")
                data, label = next(dataloader_ae)
                # push examples through the autoencoder, get latent space activations
                ex = data.to(self.device)
                ex.requires_grad=False
                self.enc.zero_grad()
                self.dec.zero_grad()
                self.ae_optim.zero_grad()
                out = self.forward(ex)
                # compute negative log probability of the example given the decoding distribution
                rec_dstr = self.rec_dstr_func(out)
                rec_loss_b = -1.*rec_dstr.log_prob(self.x_norm).mean(dim=0).sum()
                # compute the KL divergence between the encoding distribution and the prior distribution
                kl_loss_b = compute_kl_div(self.z, self.mu, torch.exp(0.5*self.log_var)).mean(dim=0).sum()
                rec_loss_agg += rec_loss_b.item()
                kl_loss_agg += kl_loss_b.item()
                (rec_loss_b + beta*kl_loss_b).backward()
                self.ae_optim.step()
            rec_loss[g] = rec_loss_agg / batch_per_group
            kl_loss[g] = kl_loss_agg / batch_per_group
            print("\tRec: {:1.4f}\tKL: {:1.4f}".format(\
                rec_loss[g], kl_loss[g]))
        return rec_loss.detach(), kl_loss.detach()
    
    def record_latent_space(self, dataset, batch_size=100, n_batches=5):
        n_elems = batch_size*n_batches
        mu_scores = torch.empty((n_elems, self.z_dim))
        std_scores = torch.empty((n_elems, self.z_dim))
        # sequential sampler
        dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False))
        i = 0
        self.eval()
        inp = None
        rec = None
        with torch.no_grad():
            while i < n_elems:
                data, label = dataloader.next()
                ex = data.to(self.device)
                if i + ex.shape[0] >= n_elems:
                    ex = ex[:n_elems-i]
                out = self.forward(ex)
                mu_scores[i:i+ex.shape[0]] = self.mu.detach()
                std_scores[i:i+ex.shape[0]] = torch.exp(0.5*self.log_var).detach()
                i += ex.shape[0]
                if inp is None:
                    inp = ex.detach()
                    rec = out.detach()
        invnorm = self.inp_inorm
        if invnorm is None:
            invnorm = InvNorm(self.inp_norm)
        rec = invnorm(rec)
        return mu_scores, std_scores, inp, rec
    

# From https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class B_TCVAE(VAE):
    
    def __init__(self, inp_norm, enc, dec, device, z_dim, inp_inorm, rec_dstr='gaussian'):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec
        self.device = device
        self.z_dim = z_dim
        self.inp_norm = inp_norm
        self.inp_inorm = inp_inorm
        
        self.ae_optim = None
        if rec_dstr == 'gaussian':
            self.rec_dstr_func = normal_dist # was a lambda, changed it to function for pickling
        elif rec_dstr == 'bernoulli':
            self.rec_dstr_func = bernoulli_dist
        else:
            raise AttributeError("expected gaussian or bernoulli, but got {}".format(rec_dstr))
        self.to(device)
        
    # FROM RTQ CHEN at https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py -----------
    def qz_estimate(self, _z_s, _z_mu, _z_log_var, _dataset_size):
        """
        _z_s: samples of z of shape (batch_size, z_dim)
        _z_mu: mu parameter of encoded distribution of shape (batch_size, z_dim)
        _z_log_var: variance parameter of encoded distribution of shape (batch_size, z_dim)
        _dataset_size: len of dataset
        """
        M = _z_s.shape[0]
        # iterate through the sample dimension
        running_sum = 0.
        for i in range(M):
            _z_s_expand = _z_s[i].unsqueeze(0).expand(_z_s.shape)
            # now compute log prob against mu, log_var
            log_probs = compute_log_prob(_z_s_expand, _z_mu, torch.exp(_z_log_var*0.5))
            running_sum += logsumexp(log_probs)
        running_sum /= M
        running_sum -= torch.log(M*_dataset_size)
        return running_sum
    
    def qz_estimate_rtqc(self, _z_s, _z_mu, _z_log_var, dataset_size):
        batch_size = _z_s.shape[0]
        _z_var = torch.exp(_z_log_var*0.5)
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = compute_log_prob(_z_s.view(batch_size, 1, self.z_dim), \
                                  _z_mu.view(1, batch_size, self.z_dim), _z_var.view(1, batch_size, self.z_dim))
        # minibatch weighted sampling
        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        return logqz_prodmarginals, logqz
    
    # ----------------------------------------
        
    def fit(self, dataset, n_group, batch_per_group=10, lr=0.001, beta=1.0,\
              batch_size=100, generator_ae=None):
        self.train()
        dataset_size = len(dataset)
        # create loss objects and optimizers
        if self.ae_optim is None:
            self.init_optim_objects(lr)
        # set up loss storage
        rec_loss = torch.zeros(n_group)
        kl_loss = torch.zeros(n_group)
        # define samplers for the AE
        n_samples = batch_size*batch_per_group*n_group
        random_sampler_ae = torch.utils.data.RandomSampler(dataset, \
                      replacement=True, num_samples=n_samples, generator=generator_ae)
        batch_sampler_ae = torch.utils.data.BatchSampler(random_sampler_ae, batch_size=batch_size, drop_last=False)
        dataloader_ae = iter(torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler_ae))
        
        for g in range(n_group):
            rec_loss_agg = 0.
            kl_loss_agg = 0.
            for b in range(batch_per_group):
                print("\rGroup: {}\t{:2.0f}%".format(g, 100*(b+1)/batch_per_group), end="")
                data, label = next(dataloader_ae)
                # push examples through the autoencoder, get latent space activations
                ex = data.to(self.device)
                ex.requires_grad=False
                self.enc.zero_grad()
                self.dec.zero_grad()
                self.ae_optim.zero_grad()
                out = self.forward(ex)
                # compute negative log probability of the example given the decoding distribution
                rec_dstr = self.rec_dstr_func(out)
                rec_loss_b = -1.*rec_dstr.log_prob(self.x_norm).mean(dim=0).sum()
                # compute the decomposed KL divergence between the encoding distribution and the prior distribution
                logqz_condx = compute_log_prob(self.z, self.mu, torch.exp(self.log_var*0.5)).sum(1)
                logpz = compute_log_prob(self.z).sum(1)
                logqz_prodmarginals, logqz = self.qz_estimate_rtqc(self.z, self.mu, self.log_var, dataset_size)
                mi_term = (logqz_condx - logqz)
                tc_term = beta * (logqz - logqz_prodmarginals)
                skl_term = (logqz_prodmarginals - logpz)
                kl_loss_b = (mi_term + tc_term + skl_term).mean()
                rec_loss_agg += rec_loss_b.item()
                kl_loss_agg += kl_loss_b.item()
                (rec_loss_b + kl_loss_b).backward()
                self.ae_optim.step()
            rec_loss[g] = rec_loss_agg / batch_per_group
            kl_loss[g] = kl_loss_agg / batch_per_group
            print("\tRec: {:1.4f}\tKL: {:1.4f}".format(\
                rec_loss[g], kl_loss[g]))
        return rec_loss.detach(), kl_loss.detach()
    
def fake_sample(z, device): # shuffle everything to get the marginals
        shuff_inds = [torch.randperm(z.shape[0]).to(device) for _ in range(z.shape[1])]

        z_fake = torch.empty_like(z)
        for _z_ind in range(z.shape[1]):
            z_fake[..., _z_ind] = z[..., _z_ind][shuff_inds[_z_ind]]
        
        return z_fake
    
_fvae_eps = 1e-8
