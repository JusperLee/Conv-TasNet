import torch
from itertools import permutations


def SI_SNR(_s, s, zero_mean=True):
    '''
         Calculate the SNR indicator between the two audios. 
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value 
    '''
    if zero_mean:
        _s = _s - torch.mean(_s)
        s = s - torch.mean(s)
    s_target = sum(torch.mul(_s, s))*s/torch.pow(torch.norm(s, p=2), 2)
    e_noise = _s - s_target
    return 20*torch.log10(torch.norm(s_target, p=2)/torch.norm(e_noise, p=2))


def permute_SI_SNR(_s_lists, s_lists):
    '''
        Calculate all possible SNRs according to 
        the permutation combination and 
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([SI_SNR(_s, s) for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
    return max(results)


def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def si_snr_loss(ests, egs):
    # spks x n x S
    refs = egs["ref"]
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
        # average the value

    # P x N
    N = egs["mix"].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N


if __name__ == "__main__":
    a_t = torch.tensor([1, 2, 3], dtype=torch.float32)
    b_t = torch.tensor([1, 4, 6], dtype=torch.float32)
    print(permute_SI_SNR([a_t], [b_t]))
