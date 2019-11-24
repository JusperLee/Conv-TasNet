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
    s_target = sum(torch.mul(_s,s))*s/torch.pow(torch.norm(s,p=2),2)
    e_noise = _s - s_target
    return 20*torch.log10(torch.norm(s_target,p=2)/torch.norm(e_noise,p=2))

def permute_SI_SNR(_s_lists,s_lists):
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
        result = sum([SI_SNR(_s,s) for _s,s in zip(_s_lists,s_list)])/length
        results.append(result)
    return max(results)

if __name__ == "__main__":
    a_t = torch.tensor([1,2,3],dtype=torch.float32)
    b_t = torch.tensor([1,4,6],dtype=torch.float32)
    print(permute_SI_SNR([a_t],[b_t]))