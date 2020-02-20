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
import sys
import editdistance
import numpy as np
import torch as th
from torch.autograd import Function
import torch.nn.functional as F

#import pykaldi related modules                            
import kaldi.fstext as kaldi_fst                           
import kaldi.hmm as kaldi_hmm                              
import kaldi.matrix as kaldi_matrix                        
import kaldi.cudamatrix as kaldi_cudamatrix 
import kaldi.lat as kaldi_lat                              
import kaldi.decoder as kaldi_decoder                      
import kaldi.chain as kaldi_chain                                                 

class MMIFunction(Function):

    """
        Args:
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
        scale = kaldi_fst.utils.lattice_scale(1.0, 0.2)
        kaldi_fst.utils.scale_lattice(scale, lattice)
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

class TeacherStudentMMI(Function):

    """
        Args:
        loglikes_T: log likelihoods from the teacher model. Note: the log-priror should be substracted 
        loglikes_S: log likelihoods from the student model. 
        asr_decoder: To run on-the-fly lattice generation
    """

    @staticmethod
    def forward(ctx, loglikes_T, loglikes_S, asr_decoder):

       # We can use either ther teacher model or the student model to generate the lattice
        decode_out = asr_decoder.decode(kaldi_matrix.Matrix(loglikes_T.detach().cpu().numpy()))
        lattice = decoder_out["lattice"]
        kaldi_lat.functions.top_sort_lattice_if_needed(lattice)
        lat_like_T, post_T, acoustic_like_T = kaldi_lat.functions.lattice_forward_backward(lattice)
        
        decodable = kaldi_decoder.DecodableMatrixScaled(loglikes_S, 1.0)
        if kaldi_lat.functions.rescore_lattice(decodable, lattice):
            lat_like_S, post_S, acoustic_like_S = kaldi_lat.functions.lattice_forward_backward(lattice)
        else:
            sys.stderr.write('ERROR: Rescore lattice failed!')
            sys.exit(0) 

        post_T = kaldi_hmm.Posterior.from_posteriors(post_T)
        post_S = kaldi_hmm.Posterior.from_posteriors(post_S)
        post_mat = post_T.to_pdf_matrix(trans_model).numpy() - post_S.to_pdf_matrix(trans_model).numpy()

        ctx.save_for_backward(th.from_numpy(post_mat).cuda())
        
        loss = F.cross_entropy(th.from_numpy(post_T), th.from_nompy(post_S))
       
        return loss.item()

    @staticmethod
    def backward(ctx, grad_out):

        grad_input, = ctx.saved_tensors

        return th.autograd.Variable(grad_input), None, None, None

class sMBRFunction(Function):
    """
        Args:
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

        return th.tensor(lat_like)

    @staticmethod
    def backward(ctx, grad_out):

        grad_input, = ctx.saved_tensors

        #flip the sign to maximize the frame accuracy 
        grad_input *= -1.0
        return th.autograd.Variable(grad_input), None, None, None, None, None

class MWEFunction(Function):
    """
        Args:
        loglikes: log likelihoods from the nnet by forwarding the input data.
                  Note, the log-prior should be substracted.
        asr_decoder: To run on-the-fly lattice generation
        trans_model: hmm transition model
        supervision: word-level transcription ids
        config: configuration for MWE loss
    """
    @staticmethod
    def forward(ctx, loglikes, asr_decoder, trans_model, supervision, config):

        decode_out = asr_decoder.decode(kaldi_matrix.Matrix(loglikes.detach().cpu().numpy()))
        lattice = decode_out["lattice"]
        kaldi_lat.functions.top_sort_lattice_if_needed(lattice)
        scale = kaldi_fst.utils.lattice_scale(config['lm_weight'], config['am_weight'])
        kaldi_fst.utils.scale_lattice(scale, lattice)
        lat_like, post, acoustic_like = kaldi_lat.functions.lattice_forward_backward(lattice)

        if config['phone_level']:
            kaldi_lat.functions.convert_lattice_to_phones(trans_model, lattice)
            _, supervision_phones = kaldi_hmm.split_to_phones(trans_model, supervision)
            supervision = [trans_model.transition_id_to_phone(cluster[0]) for cluster in supervision_phones]

        #lattice = kaldi_fst.utils.convert_compact_lattice_to_lattice(lattice)
        ifst = kaldi_fst.utils.convert_lattice_to_std(lattice)

        length = loglikes.size(0)
        ilabels = list()
        olabels = list()
        edit_distance = list()
        weights = list()
        if config['rand_path']:
            #ofst = kaldi_fst.randgen(fst, npath=8, seed=None, select='uniform', max_length=length, weighted=True, remove_total_weight=False)
        #for i in range(config['num_paths']):
            n = 0
            while len(olabels) < config['num_paths'] and n < 500:
                n += 1
                ofst = kaldi_fst.StdVectorFst()
                randint = np.random.randint(0, 10000) 
                if(kaldi_fst.utils.equal_align(ifst, length, randint, ofst)):
                    ilabel, olabel, weight = kaldi_fst.utils.get_linear_symbol_sequence(ofst)
                    if olabel not in olabels:
                        olabels.append(olabel)
                        ilabels.append(ilabel)
                        weights.append(weight.value)
                        edit_distance.append(editdistance.eval(olabel, supervision))
        else:
            nbest_lats = kaldi_fst.utils.nbest_as_fsts(ifst, config['num_paths'])
            for path in nbest_lats:
                ilabel, olabel, weight = kaldi_fst.utils.get_linear_symbol_sequence(path)
                if olabel not in olabels:
                    olabels.append(olabel)
                    ilabels.append(ilabel)
                    weights.append(weight.value)
                    edit_distance.append(editdistance.eval(olabel, supervision))

        if(config['equal_weight']):
            normalizer = th.ones(len(weights), dtype=th.float32) * 1.0/len(weights)
        else:
            normalizer = F.softmax(th.FloatTensor(weights)*-1, dim=0) 

        mean_err = th.FloatTensor(edit_distance) * normalizer
        loss = mean_err.sum()
        grad_value = (th.FloatTensor(edit_distance) - loss) * normalizer
        # compute gradients
        grad_out = th.zeros(loglikes.size())
   
        for idx in range(len(ilabels)):
            ilabel = ilabels[idx]
            for i in range(len(ilabel)):
                pdf_id = trans_model.transition_id_to_pdf(ilabel[i])
                grad_out[i][pdf_id] += grad_value[idx]
  
        ctx.save_for_backward(grad_out.cuda())

        return loss 

    def backward(ctx, grad_out):

        grad_input, = ctx.saved_tensors

        return th.autograd.Variable(grad_input), None, None, None, None

class ChainObjtiveFunction(Function):
    """
        Args:
        loglikes: log-likelihoods from the nnet after the forward operation
        den_graph: the denominator graph for chain model training
        supervision: in the format of Kaldi supervision class
        chain_opts: options for chain model training
    """

    @staticmethod
    def forward(ctx, loglikes, den_graph, supervision, chain_opts):

        loglikes = kaldi_matrix.Matrix(loglikes.detach().cpu().numpy())
        
        #if kaldi_cudamatrix.cuda_available():
        #    from kaldi.cudamatrix import CuDevice
        #    CuDevice.instantiate().select_gpu_id('yes')
        #     CuDevice.instantiate().allow_multithreading()
        nnet_out = kaldi_cudamatrix.CuMatrix().from_matrix(loglikes)
        grad = kaldi_cudamatrix.CuMatrix().from_size(nnet_out.num_rows(), nnet_out.num_cols())
        grad_xent = kaldi_cudamatrix.CuMatrix().from_size(nnet_out.num_rows(), nnet_out.num_cols())

        loss = kaldi_chain.compute_chain_objf_and_deriv(chain_opts, den_graph, supervision, nnet_out, grad, grad_xent)

        grad.add_mat(chain_opts.xent_regularize, grad_xent)
        grad_out = kaldi_matrix.Matrix(nnet_out.num_rows(), nnet_out.num_cols())
        grad.copy_to_mat(grad_out)

        ctx.save_for_backward(th.from_numpy(grad_out.numpy()).cuda())

        return th.tensor(loss[0])

    @staticmethod
    def backward(ctx, grad_out):

        grad_input, = ctx.saved_tensors
        grad_input *= -1.0
        return th.autograd.Variable(grad_input), None, None, None

