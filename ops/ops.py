"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import torch as th
from torch.autograd import Function

#import pykaldi related modules                            
import kaldi.fstext as kaldi_fst                           
import kaldi.hmm as kaldi_hmm                              
import kaldi.matrix as kaldi_matrix                        
import kaldi.lat as kaldi_lat                              
import kaldi.decoder as kaldi_decoder                      
                                                

class MMIFunction(Function):

    """
        Input:
        loglikes: log likelihoods from the nnet by forwarding the input data.
                  Note, the log-prior should be substracted.
        asr_decoder: To run on-the-fly lattice generation
        trans_model: hmm transition model
        trans_ids:   alignments in the form of hmm transition ids
        
    """
    @staticmethod
    def forward(ctx, loglikes, asr_decoder, trans_model, trans_ids):

        decode_out = asr_decoder.decode(kaldi_matrix.Matrix(loglikes.detach().cpu().numpy()))
        lattice = decode_out["lattice"]
        kaldi_lat.functions.top_sort_lattice_if_needed(lattice)
        lat_like, post = kaldi_lat.functions.lattice_forward_backward_mmi(trans_model, lattice, trans_ids, True, False, True)
        post = kaldi_hmm.Posterior.from_posteriors(post)
        post_mat = post.to_pdf_matrix(trans_model).numpy()
 
        ctx.save_for_backward(th.from_numpy(post_mat).cuda())
        #print(post_mat)
        return th.tensor(lat_like)
     
    @staticmethod
    def backward(ctx, grad_out):
     
        grad_input, = ctx.saved_tensors
        
        #flip the sign as we maximize the mutual information
        grad_input *= -1.0
        return th.autograd.Variable(grad_input), None, None, None


class sMBRFunction(Function):
    """
        Input:
        loglikes: log likelihoods from the nnet by forwarding the input data.
                  Note, the log-prior should be substracted.
        asr_decoder: To run on-the-fly lattice generation
        trans_model: hmm transition model
        trans_ids:   alignments in the form of hmm transition ids
        criterion: "smbr" or "mpfe"
        silence_phones: slience phone indexes, in the form of list of int
    """
    @staticmethod
    def forward(ctx, loglikes, asr_decoder, trans_model, trans_ids, criterion, silence_phones):

        decode_out = asr_decoder.decode(kaldi_matrix.Matrix(loglikes.detach().cpu().numpy()))
        lattice = decode_out["lattice"]
        kaldi_lat.functions.top_sort_lattice_if_needed(lattice)
        lat_like, post = kaldi_lat.functions.lattice_forward_backward_mpe_variants(trans_model, 
                                                                                  silence_phones, 
                                                                                  lattice, 
                                                                                  trans_ids, 
                                                                                  criterion, 
                                                                                  True)
        post = kaldi_hmm.Posterior.from_posteriors(post)
        post_mat = post.to_pdf_matrix(trans_model).numpy()

        ctx.save_for_backward(th.from_numpy(post_mat).cuda())
        #print(post_mat)
        return th.tensor(lat_like)

    @staticmethod
    def backward(ctx, grad_out):

        grad_input, = ctx.saved_tensors

        #flip the sign to maximize the frame accuracy 
        grad_input *= -1.0
        return th.autograd.Variable(grad_input), None, None, None, None, None

class ChainObjtiveFunction(Function):
    """
        Input:
        loglikes:
    """

    @staticmethod
    def forward(ctx, loglikes, den_graph, supervision, chain_opts)

            loglikes = kaldi_matrix.Matrix(loglikes.detach().cpu().numpy())
            nnet_out = kaldi_cudamatrix.CuMatrix().from_matrix(loglikes)
            grad = kaldi_cudamatrix.CuMatrix().from_size(nnet_out.num_rows(), nnet_out.num_cols())
            grad_xent = kaldi_cudamatrix.CuMatrix().from_size(nnet_out.num_rows(), nnet_out.num_cols())

            loss = kaldi_chain.compute_chain_objf_and_deriv(chain_opts, den, supervision, nnet_out, grad, grad_xent)

            grad.add_mat(chain_opts.xent_regularize, grad_xent)
            grad_out = kaldi_matrix.Matrix(nnet_out.num_rows(), nnet_out.num_cols())
            grad.copy_to_mat(grad_out)

            ctx.save_for_backward(th.from_numpy(grad_out).cuda())

            return th.tensor(loss[1])

    @staticmethod
    def backward(ctx, grad_out):

        grad_input, = ctx.saved_tensors

        return th.audograd.Variable(grad_input), None, None, None

