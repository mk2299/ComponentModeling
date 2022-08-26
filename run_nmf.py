import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import nimfa



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Bayesian NMF algorithm')
    parser.add_argument('--subj', default=None, type=int, help='Subject number')
    parser.add_argument('--n_comps', default=20, type=int, help='Number of components inferred using BIC')
    parser.add_argument('--data_dir', default='./ventral_visual_data/', type=str, help='Directory where the preprocessed NSD data is stored')
    parser.add_argument('--save_dir', default='./results/', type=str, help='Directory for saving the NMF outputs')
    
    args = parser.parse_args()
    subj = args.subj
    n_components = args.n_comps
    data_dir = args.data_dir 
    save_dir = args.save_dir 
    
    ### Extract responses from all ventral visual stream voxels (10,000 x num_voxels matrix) 
    data = np.load('%s/subj%d.npy'% (data_dir, subj), mmap_mode = 'r')
    

    ### Perform baseline shift to make all entries non-negative
    V = (data - data.min(0)).T
    
    ### Run 50 iterations of the Bayesian NMF algorithm
    
    for r_ in range(50):
        bdnmf = nimfa.Bd(V, seed="random_c", rank=n_components, max_iter=12, alpha=np.zeros((V.shape[0], n_components)),
          beta=np.zeros((n_components, V.shape[1])), theta=.0, k=.0, sigma=1., skip=100, stride=1,
          n_w=np.zeros((n_components, 1)), n_h=np.zeros((n_components, 1)), n_run = 1, n_sigma=False) 
        bdnmf_fit = bdnmf()

        print('Rss: %5.4f' % bdnmf_fit.fit.rss())
        print('Evar: %5.4f' % bdnmf_fit.fit.evar())
        print('K-L divergence: %5.4f' % bdnmf_fit.distance(metric='kl'))
        print('Sparseness, W: %5.4f, H: %5.4f' % bdnmf_fit.fit.sparseness())

        W = bdnmf_fit.basis()
        H = bdnmf_fit.coef()

        
        data_transformed = np.asarray(H.T).copy()
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        np.save('%s/subj%d_ncomp_%d_run%d.npy' % (save_dir, subj, n_components, r_), {'data_transformed': data_transformed, 'W': np.asarray(W), 'fit': bdnmf_fit.fit})





