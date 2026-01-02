
#! /nfs/amino-home/liyangum/miniconda3/bin/python
import numpy
import torch
import torch.autograd as autograd
import numpy as np 

import random
import Cubic, Potential
import operations
import os, json, sys

import a2b, rigid
import torch.optim as opt
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
from scipy.optimize import fmin_l_bfgs_b,fmin_cg,fmin_bfgs
from scipy.optimize import minimize
import lbfgs_rosetta
import pickle
import shutil

torch.manual_seed(6)
torch.set_num_threads(4)
np.random.seed(9)
random.seed(9)

Scale_factor = 1.0
USEGEO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readconfig(configfile=''):
    config=[]
    expdir=os.path.dirname(os.path.abspath(__file__))
    if configfile=='':
        configfile=os.path.join(expdir,'lib','ddf.json')
    config=json.load(open(configfile,'r'))
    return config 

    
class Structure:
    def __init__(self, fastafile, geofiles, foldconfig, saveprefix):
        # Load Configuration and Inputs
        self.config = readconfig(foldconfig)
        self.seqfile = fastafile
        self.foldconfig = foldconfig
        self.geofiles = geofiles

        # Load Model Results
        self.rets = [pickle.load(open(refile, 'rb')) for refile  in geofiles]
        
        # Extract Coordinates
        self.txs = []
        for ret in self.rets:
            self.txs.append(torch.from_numpy(ret['coor']).double().to(device))
        
        # Handle Geometrical Data
        self.handle_geo()

        # Extract pLDDT Scores
        self.pair = []
        for ret in self.rets:
            self.pair.append( torch.from_numpy(ret['plddt']).double().to(device))
        
        # Store Output and Sequence Info
        self.saveprefix = saveprefix
        self.seq = open(fastafile).readlines()[1].strip()
        self.L = len(self.seq)
        
        # Load Reference Arrays for Structure Construction
        basenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'base.npy'))
        self.basex = operations.Get_base(self.seq, basenpy).double().to(device)
        
        othernpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'other2.npy'))
        self.otherx = operations.Get_base(self.seq, othernpy).double().to(device)
        
        sidenpy = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'side.npy'))
        self.sidex = operations.Get_base(self.seq, sidenpy).double().to(device)        
        
        # Initialize Masks, Parameters, and FAPE
        self.init_mask()
        self.init_paras()
        self._init_fape()
    

    def _init_fape(self):
        self.tx2ds = []
        for tx in self.txs:
            true_rot, true_trans = operations.Kabsch_rigid(self.basex, tx[:, 0], tx[:, 1], tx[:, 2])
            true_x2 = tx[:, None, :, :] - true_trans[None, :, None, :]
            true_x2 = torch.einsum('ijnd,jde->ijne', true_x2, true_rot.transpose(-1,-2))
            self.tx2ds.append(true_x2)
    

    def handle_geo(self):
        oldkeys = ['dist_p', 'dist_c', 'dist_n']
        newkeys = ['pp', 'cc', 'nn']
        self.geos = []
        geo = {'pp':0, 'cc':0, 'nn':0}
        
        for ret in self.rets:    
            for nk, ok in zip(newkeys, oldkeys):
                geo[nk] = geo[nk] + (ret[ok].astype(np.float64) /(len(self.rets)))
        self.geos.append(geo)


    def init_mask(self):
        halfmask=np.zeros([self.L,self.L])
        fullmask=np.zeros([self.L,self.L])
        for i in range(self.L):
            for j in range(i+1,self.L):
                halfmask[i,j]=1
                fullmask[i,j]=1
                fullmask[j,i]=1
        self.halfmask=torch.DoubleTensor(halfmask) > 0.5
        self.fullmask=torch.DoubleTensor(fullmask) > 0.5
        self.clash_mask = torch.zeros([self.L,self.L,22,22])
        for i in range(self.L):
            for j in range(i+1,self.L):
                self.clash_mask[i,j]=1

        for i in range(self.L):
             self.clash_mask[i,i,:6,7:]=1

        for i in range(self.L-1):
            self.clash_mask[i,i+1,:,0]=0
            self.clash_mask[i,i+1,0,:]=0
            self.clash_mask[i,i+1,:,5]=0
            self.clash_mask[i,i+1,5,:]=0

        self.side_mask = rigid.side_mask(self.seq)
        self.side_mask = self.side_mask[:,None,:,None] * self.side_mask[None,:,None,:]
        self.clash_mask = (self.clash_mask > 0.5) * (self.side_mask > 0.5)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = torch.DoubleTensor(geo['cc'][:,:,-1]) < 0.5
            confimask_pp = torch.DoubleTensor(geo['pp'][:,:,-1]) < 0.5
            confimask_nn = torch.DoubleTensor(geo['nn'][:,:,-1]) < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)

        # Move masks and confimasks to the GPU/CPU device
        self.halfmask = self.halfmask.to(device)
        self.fullmask = self.fullmask.to(device)
        self.clash_mask = self.clash_mask.to(device)
        self.side_mask = self.side_mask.to(device)
        # geo_confimasks are lists
        self.geo_confimask_cc = [m.to(device) for m in self.geo_confimask_cc]
        self.geo_confimask_pp = [m.to(device) for m in self.geo_confimask_pp]
        self.geo_confimask_nn = [m.to(device) for m in self.geo_confimask_nn]


    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        self.cs_coefs = {'cc': [], 'pp': [], 'nn': []}
        self.cs_knots = {'cc': [], 'pp': [], 'nn': []}
        for geo in self.geos:
            cc_cs, cc_decs = Cubic.dis_cubic(geo['cc'], 2, 40, 36)
            pp_cs, pp_decs = Cubic.dis_cubic(geo['pp'], 2, 40, 36)
            nn_cs, nn_decs = Cubic.dis_cubic(geo['nn'], 2, 40, 36)
            self.geo_cc.append([cc_cs, cc_decs])
            self.geo_pp.append([pp_cs, pp_decs])
            self.geo_nn.append([nn_cs, nn_decs])
            L = self.L
            cc_coefs_np = np.stack([[cc_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            cc_knots_np = np.stack([[cc_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['cc'].append(torch.from_numpy(cc_coefs_np).to(device))
            self.cs_knots['cc'].append(torch.from_numpy(cc_knots_np).to(device))
            pp_coefs_np = np.stack([[pp_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            pp_knots_np = np.stack([[pp_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['pp'].append(torch.from_numpy(pp_coefs_np).to(device))
            self.cs_knots['pp'].append(torch.from_numpy(pp_knots_np).to(device))
            nn_coefs_np = np.stack([[nn_cs[i,j].c for j in range(L)] for i in range(L)], axis=0)
            nn_knots_np = np.stack([[nn_cs[i,j].x for j in range(L)] for i in range(L)], axis=0)
            self.cs_coefs['nn'].append(torch.from_numpy(nn_coefs_np).to(device))
            self.cs_knots['nn'].append(torch.from_numpy(nn_knots_np).to(device))
     

    def _cubic_pair_energy(self, atom_map, geo_cs, geo_confimask, weight_key):
        """General cubic-spline energy for CC/PP/NN pairs."""
        min_dis, max_dis, bin_num = 2, 40, 36
        dev = atom_map.device
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        lower_th = 2.5
        total = torch.zeros((), device=dev, dtype=torch.double)
        spline_key = weight_key.split('_')[1]
        coeffs_list = self.cs_coefs[spline_key]
        knots_list = self.cs_knots[spline_key]
        for block_idx, mask_block in enumerate(geo_confimask):
            mask = (atom_map <= upper_th) & mask_block & self.fullmask & (atom_map >= lower_th)
            idx = mask.nonzero(as_tuple=True)
            if idx[0].numel() > 1:
                coef = coeffs_list[block_idx][idx]
                knots = knots_list[block_idx][idx]
                part1 = Potential.cubic_distance(atom_map[mask], coef, knots, min_dis, max_dis, bin_num).sum() * self.config[weight_key] * 0.5
            else:
                part1 = torch.zeros((), device=dev, dtype=torch.double)
            part2 = ((atom_map <= lower_th) & mask_block & self.fullmask).sum() * self.config[weight_key]
            total = total + part1 + part2
        return total

    # GPU-friendly torsion and angle energy helpers
    def _cubic_torsion_energy(self, atom_map, coef, x_vals, weight_key, num_bin):
        energy = Potential.cubic_torsion(atom_map, coef, x_vals, num_bin)
        return energy.sum() * self.config[weight_key]

    def _cubic_angle_energy(self, atom_map, coef, x_vals, weight_key, num_bin):
        energy = Potential.cubic_angle(atom_map, coef, x_vals, num_bin)
        return energy.sum() * self.config[weight_key]

    def compute_cc_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,1], coor[:,1])
        return self._cubic_pair_energy(atom_map, self.geo_cc, self.geo_confimask_cc, 'weight_cc')

    def compute_pp_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,0], coor[:,0])
        return self._cubic_pair_energy(atom_map, self.geo_pp, self.geo_confimask_pp, 'weight_pp')

    def compute_nn_energy(self, coor):
        atom_map = operations.pair_distance(coor[:,-1], coor[:,-1])
        return self._cubic_pair_energy(atom_map, self.geo_nn, self.geo_confimask_nn, 'weight_nn')

    def compute_pccp_energy(self, coor):
        # P-C-C-P dihedral energy on GPU
        p = coor[:, 0]
        c = coor[:, 1]
        dia = operations.dihedral(
            p[self.pccpi], c[self.pccpi], c[self.pccpj], p[self.pccpj]
        )
        return self._cubic_torsion_energy(dia, self.pccp_coe, self.pccp_x, 'weight_pccp', 36)

    def compute_cnnc_energy(self, coor):
        # C-N-N-C dihedral energy on GPU
        n = coor[:, -1]
        c = coor[:, 1]
        dia = operations.dihedral(
            c[self.cnnci], n[self.cnnci], n[self.cnncj], c[self.cnncj]
        )
        return self._cubic_torsion_energy(dia, self.cnnc_coe, self.cnnc_x, 'weight_cnnc', 36)

    def compute_pnnp_energy(self, coor):
        # P-N-N-P dihedral energy on GPU
        n = coor[:, -1]
        p = coor[:, 0]
        dia = operations.dihedral(
            p[self.pnnpi], n[self.pnnpi], n[self.pnnpj], p[self.pnnpj]
        )
        return self._cubic_torsion_energy(dia, self.pnnp_coe, self.pnnp_x, 'weight_pnnp', 36)

    def compute_pcc_energy(self, coor):
        # P-C-C angle energy on GPU
        p = coor[:, 1]
        c = coor[:, 2]
        ang = operations.angle(
            p[self.pcci], c[self.pcci], c[self.pccj]
        )
        return self._cubic_angle_energy(ang, self.pcc_coe, self.pcc_x, 'weight_pcc', 12)

    def compute_fape_energy(self,coor,ep=1e-3,epmax=20):
        energy= 0
        for tx in self.tx2ds:
            px_mean = coor[:,[1]]
            p_rot   = operations.rigidFrom3Points(coor)
            p_tran  = px_mean[:,0]
            pred_x2 = coor[:,None,:,:] - p_tran[None,:,None,:] # Lx Lrot N , 3
            pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep )
            energy = energy + torch.sum(  torch.clamp(errmap,max=epmax)        )
        return energy * self.config['weight_fape']

    def compute_bond_energy(self,coor,other_coor):
        # 3.87
        o3 = other_coor[:-1,-2]
        p  = coor[1:,0]
        dis = (o3-p).norm(dim=-1)
        energy = ((dis-1.607)**2).sum()
        return energy * self.config['weight_bond']

    def tooth_func(self,errmap, ep = 0.05):
        return -1/(errmap/10+ep) + (1/ep)
    
    def reweight_func(self,ww):
        reweighting = torch.pow(ww,self.config['pair_weight_power'])
        reweighting[ww < self.config['pair_weight_min']] = 0
        return reweighting
    
    def compute_fape_energy_fromquat(self,x,coor,ep=1e-6,epmax=100):
        energy= 0
        p_rot,px_mean = a2b.Non2rot(x[:,:9],x.shape[0]),x[:,9:]
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse

        for tx,weightplddt in zip(self.tx2ds,self.pair):
            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep ) 
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']
    
    def compute_fape_energy_fromcoor(self,coor,ep=1e-6,epmax=100):
        energy= 0
        
        p_rot,px_mean = operations.Kabsch_rigid(self.basex,coor[:,0],coor[:,1],coor[:,2])
        pred_x2 = coor[:,None,:,:] - px_mean[None,:,None,:] # Lx Lrot N , 3
        pred_x2 = torch.einsum('ijnd,jde->ijne',pred_x2,p_rot.transpose(-1,-2)) # transpose should be equal to inverse
        
        for tx,weightplddt in zip(self.tx2ds,self.pair):
            tamplate_dist_map = torch.min( tx.norm(dim=-1), dim=2   )[0]
            errmap=torch.sqrt( ((pred_x2 - tx)**2).sum(dim=-1) + ep ) 
            energy = energy + torch.sum( ( (torch.clamp(errmap,max=self.config['FAPE_max'])**self.config['pair_error_power'])  * self.reweight_func(weightplddt[...,None]) )[tamplate_dist_map>self.config['pair_rest_min_dist']]    )

        return energy * self.config['weight_fape']
    
    
    def energy(self, rama):
        coor = a2b.quat2b(self.basex, rama[:, 9:])
        other_coor = a2b.quat2b(self.otherx, rama[:, 9:])
        side_coor = a2b.quat2b(self.sidex, torch.cat([rama[:, :9], coor[:, -1]], dim=-1))

        E_cc = self.compute_cc_energy(coor) / len(self.geofiles) if self.config['weight_cc'] > 0 else 0
        E_pp = self.compute_pp_energy(coor) / len(self.geofiles) if self.config['weight_pp'] > 0 else 0
        E_nn = self.compute_nn_energy(coor) / len(self.geofiles) if self.config['weight_nn'] > 0 else 0
        E_pccp = self.compute_pccp_energy(coor) / len(self.geofiles) if self.config['weight_pccp'] > 0 else 0
        E_cnnc = self.compute_cnnc_energy(coor) / len(self.geofiles) if self.config['weight_cnnc'] > 0 else 0
        E_pnnp = self.compute_pnnp_energy(coor) / len(self.geofiles) if self.config['weight_pnnp'] > 0 else 0
        E_vdw = self.compute_full_clash(coor, other_coor, side_coor) if self.config['weight_vdw'] > 0 else 0
        E_fape = self.compute_fape_energy_fromquat(rama[:, 9:], coor) / len(self.geofiles) if self.config['weight_fape'] > 0 else 0
        E_bond = self.compute_bond_energy(coor, other_coor) if self.config['weight_bond'] > 0 else 0

        return E_vdw + E_fape + E_bond + E_pp + E_cc + E_nn + E_pccp + E_cnnc + E_pnnp


    def energy_from_coor(self, coor):
        E_cc = self.compute_cc_energy(coor) if self.config['weight_cc'] > 0 else 0
        E_pp = self.compute_pp_energy(coor) if self.config['weight_pp'] > 0 else 0
        E_nn = self.compute_nn_energy(coor) if self.config['weight_nn'] > 0 else 0
        E_fape = (self.compute_fape_energy_fromcoor(coor) / len(self.geofiles)) if self.config['weight_fape'] > 0 else 0
        # print(E_fape, E_pp, E_cc, E_nn)
        return E_fape + E_pp + E_cc + E_nn 

    def obj_func_grad_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama.requires_grad=True
        if rama.grad:
            rama.grad.zero_()
        f=self.energy(rama.view(self.L,21))*Scale_factor
        grad_value=autograd.grad(f,rama)[0]
        return grad_value.data.numpy().astype(np.float64)
    
    def obj_func_np(self,rama_):
        rama=torch.DoubleTensor(rama_)
        rama=rama.view(self.L,21)
        with torch.no_grad():
            f = self.energy(rama)*Scale_factor
            return f.item()

    def saveconfig(self,dict,confile):
        json_object = json.dumps(dict, indent = 4)
        wfile = open(confile,'w')
        wfile.write(json_object)
        wfile.close()
    
    def scoring(self):
        geoscale = self.config['geo_scale']
        self.config['weight_pp'] = geoscale * self.config['weight_pp']
        self.config['weight_cc'] = geoscale * self.config['weight_cc']
        self.config['weight_nn'] = geoscale * self.config['weight_nn']
        self.config['weight_pccp'] = geoscale * self.config['weight_pccp']
        self.config['weight_cnnc'] = geoscale * self.config['weight_cnnc']
        self.config['weight_pnnp'] = geoscale * self.config['weight_pnnp']  
        
        energy_dict = {}
        saveenergy_dict  = {}
        
        with torch.no_grad():
            for retfile, tx in zip(self.geofiles, self.txs):
                one = self.energy_from_coor(tx)
                aaretfile = os.path.basename(retfile) 
                energy_dict[aaretfile] = one.item()
                saveenergy_dict[retfile] = one.item()
            self.saveconfig(energy_dict, self.saveprefix)


    def foldning(self):
        minenergy=1e16
        count=0
        for tx in self.txs:
            count+=1
        
        minirama=None

        ilter = self.init_ret
        selected_ret = self.geofiles[ilter]
        try:
            rama=self.init_quat(ilter).data.numpy()
            self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
            rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10,maxfun=100)[0]
            rama = rama.flatten()
        except:
            rama=self.init_quat_safe(ilter).data.numpy()
            self.config=readconfig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib','vdw.json'))
            rama = fmin_l_bfgs_b(func=self.obj_func_np, x0=rama,  fprime=self.obj_func_grad_np,iprint=10,maxfun=100)[0]
            rama = rama.flatten()
            
        self.config=readconfig(self.foldconfig)
        geoscale = self.config['geo_scale']
        self.config['weight_pp'] =geoscale * self.config['weight_pp']
        self.config['weight_cc'] =geoscale * self.config['weight_cc']
        self.config['weight_nn'] =geoscale * self.config['weight_nn']
        self.config['weight_pccp'] =geoscale * self.config['weight_pccp']
        self.config['weight_cnnc'] =geoscale * self.config['weight_cnnc']
        self.config['weight_pnnp'] =geoscale * self.config['weight_pnnp']
        for i in range(3):
            line_min = lbfgs_rosetta.ArmijoLineMinimization(self.obj_func_np,self.obj_func_grad_np,True,len(rama),120)
            lbfgs_opt = lbfgs_rosetta.lbfgs(self.obj_func_np,self.obj_func_grad_np)
            rama=lbfgs_opt.run(rama,256,lbfgs_rosetta.absolute_converge_test,line_min,8000,self.obj_func_np,self.obj_func_grad_np,1e-9)
        newrama=rama+0.0
        newrama=torch.DoubleTensor(newrama) 
        current_energy =self.obj_func_np(rama)

        if current_energy < minenergy:
            print(current_energy,minenergy)
            minenergy=current_energy
            self.outpdb(newrama,self.saveprefix+'.pdb',energystr=str(current_energy))


    def outpdb(self,rama,savefile,start=0,end=10000,energystr=''):
        coor_np=a2b.quat2b(self.basex,rama.view(self.L,21)[:,9:]).data.numpy()
        other_np=a2b.quat2b(self.otherx,rama.view(self.L,21)[:,9:]).data.numpy()
        shaped_rama=rama.view(self.L,21)
        coor = torch.FloatTensor(coor_np)
        side_coor_NP = a2b.quat2b(self.sidex,torch.cat([shaped_rama[:,:9],coor[:,-1]],dim=-1)).data.numpy()
        
        Atom_name=[' P  '," C4'",' N1 ']
        Other_Atom_name = [" O5'"," C5'"," C3'"," O3'"," C1'"]
        other_last_name = ['O',"C","C","O","C"]

        side_atoms=         [' N1 ',' C2 ',' O2 ',' N2 ',' N3 ',' N4 ',' C4 ',' O4 ',' C5 ',' C6 ',' O6 ',' N6 ',' N7 ',' N8 ',' N9 ']
        side_last_name =    ['N',      "C",   "O",   "N",   "N",   'N',   'C',   'O',   'C',   'C',   'O',   'N',    'N', 'N','N']

        base_dict = rigid.base_table()
        
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1

            for j in range(other_np.shape[1]):
                outs=('ATOM  ',count,Other_Atom_name[j],self.seq[i],'A',i+1,other_np[i][j][0],other_np[i][j][1],other_np[i][j][2],0,0,other_last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                    count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()
    
    
    def outpdb_coor(self,coor_np,savefile,start=0,end=1000,energystr=''):
        Atom_name=[' P  '," C4'",' N1 ']
        last_name=['P','C','N']
        wstr=[f'REMARK {str(energystr)}']
        templet='%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s'
        count=1
        for i in range(self.L):
            if self.seq[i] in ['a','g','A','G']:
                Atom_name = [' P  '," C4'",' N9 ']

            elif self.seq[i] in ['c','u','C','U']:
                Atom_name = [' P  '," C4'",' N1 ']
            
            for j in range(coor_np.shape[1]):
                outs=('ATOM  ',count,Atom_name[j],self.seq[i],'A',i+1,coor_np[i][j][0],coor_np[i][j][1],coor_np[i][j][2],0,0,last_name[j],'')
                if i>=start-1 and i < end:
                    wstr.append(templet % outs)
                count+=1
            
        wstr='\n'.join(wstr)
        wfile=open(savefile,'w')
        wfile.write(wstr)
        wfile.close()


if __name__ == '__main__': 

    fastafile = sys.argv[1]
    foldconfig = sys.argv[2]
    save_prefix = sys.argv[3]
    retfiles = sys.argv[4:]

    save_parent_dir = os.path.dirname(save_prefix)
    if not os.path.isdir(save_parent_dir):
        os.makedirs(save_parent_dir)

    retfiles.sort()
    print(retfiles)

    stru = Structure(fastafile, retfiles, foldconfig, save_prefix)    
    stru.scoring()
