#! /nfs/amino-home/liyangum/miniconda3/bin/python
import json
import os
import pickle
import random
import sys

import a2b
import Cubic
import numpy as np
import operations
import Potential
import rigid
import torch
import torch.optim as opt

# from scipy.optimize import minimize

torch.manual_seed(6)
np.random.seed(9)
random.seed(9)


Scale_factor = 1.0
USEGEO = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def readconfig(configfile=""):
    config = []
    expdir = os.path.dirname(os.path.abspath(__file__))
    if configfile == "":
        configfile = os.path.join(expdir, "lib", "ddf.json")
    config = json.load(open(configfile, "r"))
    return config


class Structure:
    def __init__(
        self, fastafile, geofiles, saveprefix, initial_ret, foldconfig, af3file=None
    ):
        self.config = readconfig(foldconfig)
        self.seqfile = fastafile
        self.init_ret = initial_ret
        self.foldconfig = foldconfig
        self.geofiles = geofiles
        self.rets = [pickle.load(open(refile, "rb")) for refile in geofiles]
        self.txs = []
        for ret in self.rets:
            self.txs.append(torch.from_numpy(ret["coor"]).double().to(device))
        self.handle_geo()
        self.pair = []
        for ret in self.rets:
            self.pair.append(torch.from_numpy(ret["plddt"]).double().to(device))
        self.saveprefix = saveprefix
        self.seq = open(fastafile).readlines()[1].strip()
        self.L = len(self.seq)
        basenpy = np.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "base.npy")
        )
        self.basex = operations.Get_base(self.seq, basenpy).to(device)
        othernpy = np.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "lib", "other2.npy"
            )
        )
        self.otherx = operations.Get_base(self.seq, othernpy).to(device)
        sidenpy = np.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "side.npy")
        )
        self.sidex = operations.Get_base(self.seq, sidenpy).to(device)

        self.init_mask()
        self.init_paras()
        self._init_fape()
        self.tx2ds = [td.to(device) for td in self.tx2ds]
        self.local_weight = torch.ones(self.L, self.L).to(device)

        for i in range(self.L):
            for j in range(i + 1, min(self.L, i + 2)):
                self.local_weight[i, j] = self.local_weight[j, i] = 4
            for j in range(i + 2, min(self.L, i + 3)):
                self.local_weight[i, j] = self.local_weight[j, i] = 3
            for j in range(i + 3, min(self.L, i + 4)):
                self.local_weight[i, j] = self.local_weight[j, i] = 2

        # Load AlphaFold3 prediction if provided
        if af3file and os.path.exists(af3file):
            self.af3_coords = self.load_af3_structure(af3file)
            print(f"[DRfold2] Loaded AlphaFold3 structure from {af3file}")
            self._init_af3()
        elif self.config.get("weight_af3", 0) > 0:
            print(
                "[DRfold2] Warning: AlphaFold3 weight is set but no AF3 file provided"
            )

    def load_af3_structure(self, pdbfile):
        """Load AlphaFold3 predicted structure from PDB file"""
        print(f"[DRfold2] Loading AlphaFold3 prediction from {pdbfile}")
        af3_coords = torch.zeros(
            (self.L, 3, 3), device=device, dtype=torch.double
        )  # L residues, 3 atoms (P, C4', N1/N9), 3 coordinates

        # Read PDB file and extract P, C4', N1/N9 coordinates
        atom_types = [" P  ", " C4'", " N1 "]
        purine_bases = ["A", "G", "a", "g"]  # noqa: F841

        with open(pdbfile, "r") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue

                atom_type = line[12:16]
                res_id = int(line[22:26]) - 1  # PDB is 1-indexed
                if res_id < 0 or res_id >= self.L:
                    continue

                # Handle purines (A, G) which have N9 instead of N1
                if atom_type == " N1 " and self.seq[res_id].upper() in ["A", "G"]:
                    continue
                if atom_type == " N9 " and self.seq[res_id].upper() in ["A", "G"]:
                    atom_idx = 2  # Index for the third atom (N1/N9)
                elif atom_type in atom_types:
                    atom_idx = atom_types.index(atom_type)
                else:
                    continue

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                af3_coords[res_id, atom_idx] = torch.tensor(
                    [x, y, z], device=device, dtype=torch.double
                )

        # Check if structure is complete
        if torch.any(torch.sum(af3_coords, dim=-1) == 0):
            print("[DRfold2] Warning: AlphaFold3 structure is incomplete")

        return af3_coords

    def _init_af3(self):
        """Initialize AlphaFold3 aligned coordinates for FAPE calculation"""
        if not hasattr(self, "af3_coords"):
            return

        self.af3_rot, self.af3_trans = operations.Kabsch_rigid(
            self.basex,
            self.af3_coords[:, 0],  # P atoms
            self.af3_coords[:, 1],  # C4' atoms
            self.af3_coords[:, 2],  # N1/N9 atoms
        )

        # Create aligned coordinates for fast energy calculation
        # Shape: [L, L, 3, 3]
        self.af3_aligned = self.af3_coords.unsqueeze(0).repeat(self.L, 1, 1, 1)

        # Translate by af3_trans
        self.af3_aligned = self.af3_aligned - self.af3_trans.unsqueeze(1).unsqueeze(1)

        # Rotate by af3_rot
        self.af3_aligned = torch.einsum(
            "ijkl,ild->ijkd", self.af3_aligned, self.af3_rot.transpose(-1, -2)
        )

        print(
            f"[DRfold2] AlphaFold3 alignment initialized "
            f"(shape: {self.af3_aligned.shape})"
        )

    def _init_fape(self):
        self.tx2ds = []
        for tx in self.txs:
            true_rot, true_trans = operations.Kabsch_rigid(
                self.basex, tx[:, 0], tx[:, 1], tx[:, 2]
            )
            true_x2 = tx[:, None, :, :] - true_trans[None, :, None, :]
            true_x2 = torch.einsum(
                "ijnd,jde->ijne", true_x2, true_rot.transpose(-1, -2)
            )
            self.tx2ds.append(true_x2)

    def handle_geo(self):
        oldkeys = ["dist_p", "dist_c", "dist_n"]
        newkeys = ["pp", "cc", "nn"]
        self.geos = []
        for ret in self.rets:
            geo = {}
            for nk, ok in zip(newkeys, oldkeys):
                geo[nk] = torch.from_numpy(ret[ok].astype(np.float64)).to(device) + 0
            self.geos.append(geo)

    def init_mask(self):
        halfmask = np.zeros([self.L, self.L])
        fullmask = np.zeros([self.L, self.L])
        for i in range(self.L):
            for j in range(i + 1, self.L):
                halfmask[i, j] = 1
                fullmask[i, j] = 1
                fullmask[j, i] = 1
        self.halfmask = (torch.DoubleTensor(halfmask) > 0.5).to(device)
        self.fullmask = (torch.DoubleTensor(fullmask) > 0.5).to(device)
        self.clash_mask = torch.zeros([self.L, self.L, 22, 22], device=device)
        for i in range(self.L):
            for j in range(i + 1, self.L):
                self.clash_mask[i, j] = 1

        for i in range(self.L):
            self.clash_mask[i, i, :6, 7:] = 1

        for i in range(self.L - 1):
            self.clash_mask[i, i + 1, :, 0] = 0
            self.clash_mask[i, i + 1, 0, :] = 0
            self.clash_mask[i, i + 1, :, 5] = 0
            self.clash_mask[i, i + 1, 5, :] = 0

        self.side_mask = rigid.side_mask(self.seq).to(device)
        self.side_mask = (
            self.side_mask[:, None, :, None] * self.side_mask[None, :, None, :]
        ).to(device)
        self.clash_mask = ((self.clash_mask > 0.5) * (self.side_mask > 0.5)).to(device)

        self.geo_confimask_cc = []
        self.geo_confimask_pp = []
        self.geo_confimask_nn = []
        for geo in self.geos:
            confimask_cc = geo["cc"][:, :, -1] < 0.5
            confimask_pp = geo["pp"][:, :, -1] < 0.5
            confimask_nn = geo["nn"][:, :, -1] < 0.5
            self.geo_confimask_cc.append(confimask_cc)
            self.geo_confimask_pp.append(confimask_pp)
            self.geo_confimask_nn.append(confimask_nn)

    def init_paras(self):
        self.geo_cc = []
        self.geo_pp = []
        self.geo_nn = []
        self.cs_coefs = {"cc": [], "pp": [], "nn": []}
        self.cs_knots = {"cc": [], "pp": [], "nn": []}
        for geo in self.geos:
            cc_cs, cc_decs = Cubic.dis_cubic(geo["cc"], 2, 40, 36)
            pp_cs, pp_decs = Cubic.dis_cubic(geo["pp"], 2, 40, 36)
            nn_cs, nn_decs = Cubic.dis_cubic(geo["nn"], 2, 40, 36)
            self.geo_cc.append([cc_cs, cc_decs])
            self.geo_pp.append([pp_cs, pp_decs])
            self.geo_nn.append([nn_cs, nn_decs])

            L = self.L
            cc_coefs_np = np.stack(
                [[cc_cs[i, j].c for j in range(L)] for i in range(L)], axis=0
            )
            cc_knots_np = np.stack(
                [[cc_cs[i, j].x for j in range(L)] for i in range(L)], axis=0
            )
            self.cs_coefs["cc"].append(torch.from_numpy(cc_coefs_np).to(device))
            self.cs_knots["cc"].append(torch.from_numpy(cc_knots_np).to(device))

            pp_coefs_np = np.stack(
                [[pp_cs[i, j].c for j in range(L)] for i in range(L)], axis=0
            )
            pp_knots_np = np.stack(
                [[pp_cs[i, j].x for j in range(L)] for i in range(L)], axis=0
            )
            self.cs_coefs["pp"].append(torch.from_numpy(pp_coefs_np).to(device))
            self.cs_knots["pp"].append(torch.from_numpy(pp_knots_np).to(device))

            nn_coefs_np = np.stack(
                [[nn_cs[i, j].c for j in range(L)] for i in range(L)], axis=0
            )
            nn_knots_np = np.stack(
                [[nn_cs[i, j].x for j in range(L)] for i in range(L)], axis=0
            )
            self.cs_coefs["nn"].append(torch.from_numpy(nn_coefs_np).to(device))
            self.cs_knots["nn"].append(torch.from_numpy(nn_knots_np).to(device))

    def compute_bb_clash(self, coor, other_coor):
        com_coor = torch.cat([coor, other_coor], dim=1)
        com_dis = (com_coor[:, None, :, None, :] - com_coor[None, :, None, :, :]).norm(
            dim=-1
        )
        dynamicmask2_vdw = (com_dis <= 3.15) * (self.clash_mask)
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw], 3.15)
        return vdw_dynamic.sum() * self.config["weight_vdw"]

    def compute_full_clash(self, coor, other_coor, side_coor):
        com_coor = torch.cat([coor[:, :2], other_coor, side_coor], dim=1)
        com_dis = (com_coor[:, None, :, None, :] - com_coor[None, :, None, :, :]).norm(
            dim=-1
        )
        dynamicmask2_vdw = (com_dis <= 2.5) * (self.clash_mask)
        vdw_dynamic = Potential.LJpotential(com_dis[dynamicmask2_vdw], 2.5)
        return vdw_dynamic.sum() * self.config["weight_vdw"]

    def _cubic_pair_energy(self, atom_map, geo_cs, geo_confimask, weight_key):
        """General cubic-spline energy for CC/PP/NN pairs."""
        min_dis, max_dis, bin_num = 2, 40, 36
        dev = atom_map.device
        upper_th = max_dis - ((max_dis - min_dis) / bin_num) * 0.5
        lower_th = 2.5
        total = torch.zeros((), device=dev, dtype=torch.double)
        spline_key = weight_key.split("_")[1]  # 'cc', 'pp', or 'nn'
        coeffs_list = self.cs_coefs[spline_key]
        knots_list = self.cs_knots[spline_key]
        for block_idx, mask_block in enumerate(geo_confimask):
            mask = (
                (atom_map <= upper_th)
                & mask_block
                & self.fullmask
                & (atom_map >= lower_th)
            )
            idx = mask.nonzero(as_tuple=True)
            if idx[0].numel() > 1:
                coef = coeffs_list[block_idx][idx]
                knots = knots_list[block_idx][idx]
                part1 = (
                    Potential.cubic_distance(
                        atom_map[mask], coef, knots, min_dis, max_dis, bin_num
                    ).sum()
                    * self.config[weight_key]
                    * 0.5
                )
            else:
                part1 = torch.zeros((), device=dev)
            part2 = (
                (atom_map <= lower_th) & mask_block & self.fullmask
            ).sum() * self.config[weight_key]
            total = total + part1 + part2
        return total

    def compute_cc_energy(self, coor):
        atom_map = operations.pair_distance(coor[:, 1], coor[:, 1])
        return self._cubic_pair_energy(
            atom_map, self.geo_cc, self.geo_confimask_cc, "weight_cc"
        )

    def compute_pp_energy(self, coor):
        atom_map = operations.pair_distance(coor[:, 0], coor[:, 0])
        return self._cubic_pair_energy(
            atom_map, self.geo_pp, self.geo_confimask_pp, "weight_pp"
        )

    def compute_nn_energy(self, coor):
        atom_map = operations.pair_distance(coor[:, -1], coor[:, -1])
        return self._cubic_pair_energy(
            atom_map, self.geo_nn, self.geo_confimask_nn, "weight_nn"
        )

    def compute_pccp_energy(self, coor):
        p_atoms = coor[:, 0]
        c_atoms = coor[:, 1]
        pccpmap = operations.dihedral(
            p_atoms[self.pccpi],
            c_atoms[self.pccpi],
            c_atoms[self.pccpj],
            p_atoms[self.pccpj],
        )
        neg_log = Potential.cubic_torsion(pccpmap, self.pccp_coe, self.pccp_x, 36)
        return neg_log.sum() * self.config["weight_pccp"]

    def compute_cnnc_energy(self, coor):
        n_atoms = coor[:, -1]
        c_atoms = coor[:, 1]
        pccpmap = operations.dihedral(
            c_atoms[self.cnnci],
            n_atoms[self.cnnci],
            n_atoms[self.cnncj],
            c_atoms[self.cnncj],
        )
        neg_log = Potential.cubic_torsion(pccpmap, self.cnnc_coe, self.cnnc_x, 36)
        return neg_log.sum() * self.config["weight_cnnc"]

    def compute_pnnp_energy(self, coor):
        n_atoms = coor[:, -1]
        p_atoms = coor[:, 0]
        pccpmap = operations.dihedral(
            p_atoms[self.pnnpi],
            n_atoms[self.pnnpi],
            n_atoms[self.pnnpj],
            p_atoms[self.pnnpj],
        )
        neg_log = Potential.cubic_torsion(pccpmap, self.pnnp_coe, self.pnnp_x, 36)
        return neg_log.sum() * self.config["weight_pnnp"]

    def compute_pcc_energy(self, coor):
        p_atoms = coor[:, 1]
        c_atoms = coor[:, 2]
        pccmap = operations.angle(
            p_atoms[self.pcci], c_atoms[self.pcci], c_atoms[self.pccj]
        )
        neg_log = Potential.cubic_angle(pccmap, self.pcc_coe, self.pcc_x, 12)
        return neg_log.sum() * self.config["weight_pcc"]

    def compute_fape_energy(self, coor, ep=1e-3, epmax=20):
        energy = 0
        for tx in self.tx2ds:
            px_mean = coor[:, [1]]
            p_rot = operations.rigidFrom3Points(coor)
            p_tran = px_mean[:, 0]
            pred_x2 = coor[:, None, :, :] - p_tran[None, :, None, :]  # Lx Lrot N , 3
            pred_x2 = torch.einsum(
                "ijnd,jde->ijne", pred_x2, p_rot.transpose(-1, -2)
            )  # transpose should be equal to inverse
            errmap = torch.sqrt(((pred_x2 - tx) ** 2).sum(dim=-1) + ep)
            energy = energy + torch.sum(torch.clamp(errmap, max=epmax))
        return energy * self.config["weight_fape"]

    def compute_bond_energy(self, coor, other_coor):
        # 3.87
        o3 = other_coor[:-1, -2]
        p = coor[1:, 0]
        dis = (o3 - p).norm(dim=-1)
        energy = ((dis - 1.607) ** 2).sum()
        return energy * self.config["weight_bond"]

    def tooth_func(self, errmap, ep=0.05):
        return -1 / (errmap / 10 + ep) + (1 / ep)

    def reweight_func(self, ww):
        reweighting = torch.pow(ww, self.config["pair_weight_power"])
        reweighting[ww < self.config["pair_weight_min"]] = 0
        return reweighting

    def compute_fape_energy_fromquat(self, x, coor, ep=1e-6, epmax=100):
        energy = 0
        p_rot, px_mean = a2b.Non2rot(x[:, :9], x.shape[0]), x[:, 9:]
        pred_x2 = coor[:, None, :, :] - px_mean[None, :, None, :]  # Lx Lrot N , 3
        pred_x2 = torch.einsum(
            "ijnd,jde->ijne", pred_x2, p_rot.transpose(-1, -2)
        )  # transpose should be equal to inverse
        for tx, weightplddt in zip(self.tx2ds, self.pair):

            tamplate_dist_map = torch.min(tx.norm(dim=-1), dim=2)[0]
            errmap = torch.sqrt(((pred_x2 - tx) ** 2).sum(dim=-1) + ep)
            energy = energy + torch.sum(
                (
                    (
                        torch.clamp(errmap, max=self.config["FAPE_max"])
                        ** self.config["pair_error_power"]
                    )
                    * self.reweight_func(weightplddt[..., None])
                    * self.local_weight[..., None]
                )[tamplate_dist_map > self.config["pair_rest_min_dist"]]
            )

        return energy * self.config["weight_fape"]

    def compute_af3_energy(self, coor, ep=1e-6):
        """Compute energy based on deviation from AlphaFold3 structure"""
        # Skip if no AF3 structure is available
        if not hasattr(self, "af3_coords"):
            return 0

        # Print status message to indicate AF3 integration is active
        # print(f"[DRfold2] Computing AlphaFold3 energy contribution "
        #       f"(weight: {self.config['weight_af3']})")

        # Calculate rigid transformation
        p_rot = operations.rigidFrom3Points(coor)
        p_trans = coor[:, 1].clone()  # Use C4' as center of transform

        # Apply transformation to coordinates
        pred_coords = coor.clone().unsqueeze(1)  # Shape: [L, 1, 3, 3]
        pred_coords = pred_coords - p_trans.unsqueeze(1).unsqueeze(1)  # Translate
        pred_coords = torch.einsum(
            "ijkl,ild->ijkd", pred_coords, p_rot.transpose(-1, -2)
        )  # Rotate

        # Calculate error map between aligned structures
        af3_aligned = self.af3_aligned.clone()  # Shape: [L, L, 3, 3]
        errmap = torch.sqrt(
            ((pred_coords - af3_aligned) ** 2).sum(dim=-1) + ep
        )  # Distance error

        # Apply error power and clamping
        max_dist = self.config.get("AF3_max", 20.0)
        error_power = self.config.get("af3_error_power", 2.0)

        # Calculate per-atom distances to filter out distant pairs
        atom_dists = torch.min(af3_aligned.norm(dim=-1), dim=2)[
            0
        ]  # Min distance between atoms

        # Get pair weighting based on distance threshold
        pair_min_dist = self.config.get("pair_rest_min_dist", 2.0)
        mask = atom_dists > pair_min_dist

        # Apply error function
        energy = torch.sum((torch.clamp(errmap, max=max_dist) ** error_power)[mask])

        return energy * self.config["weight_af3"]

    def energy(self, rama):
        coor = a2b.quat2b(self.basex, rama[:, 9:])
        other_coor = a2b.quat2b(self.otherx, rama[:, 9:])
        side_coor = a2b.quat2b(
            self.sidex, torch.cat([rama[:, :9], coor[:, -1]], dim=-1)
        )

        if self.config["weight_cc"] > 0:
            E_cc = self.compute_cc_energy(coor) / len(self.rets)
        else:
            E_cc = 0
        if self.config["weight_pp"] > 0:
            E_pp = self.compute_pp_energy(coor) / len(self.rets)
        else:
            E_pp = 0
        if self.config["weight_nn"] > 0:
            E_nn = self.compute_nn_energy(coor) / len(self.rets)
        else:
            E_nn = 0

        if self.config["weight_pccp"] > 0:
            E_pccp = self.compute_pccp_energy(coor) / len(self.rets)
        else:
            E_pccp = 0

        if self.config["weight_cnnc"] > 0:
            E_cnnc = self.compute_cnnc_energy(coor) / len(self.rets)
        else:
            E_cnnc = 0

        if self.config["weight_pnnp"] > 0:
            E_pnnp = self.compute_pnnp_energy(coor) / len(self.rets)
        else:
            E_pnnp = 0

        if self.config["weight_vdw"] > 0:
            E_vdw = self.compute_full_clash(coor, other_coor, side_coor)
        else:
            E_vdw = 0

        if self.config["weight_fape"] > 0:
            E_fape = self.compute_fape_energy_fromquat(rama[:, 9:], coor) / len(
                self.rets
            )
        else:
            E_fape = 0

        if self.config.get("weight_af3", 0) > 0 and hasattr(self, "af3_coords"):
            E_af3 = self.compute_af3_energy(coor)
        else:
            E_af3 = 0

        if self.config["weight_bond"] > 0:
            E_bond = self.compute_bond_energy(coor, other_coor)
        else:
            E_bond = 0

        return (
            E_vdw
            + E_fape
            + E_bond
            + E_pp
            + E_cc
            + E_nn
            + E_pccp
            + E_cnnc
            + E_pnnp
            + E_af3
        )

    def obj_func_grad_np(self, rama_):
        rama = torch.DoubleTensor(rama_)
        rama.requires_grad = True
        if rama.grad:
            rama.grad.zero_()
        f = self.energy(rama.view(self.L, 21)) * Scale_factor
        grad_value = torch.autograd.grad(f, rama)[0]
        return grad_value.data.numpy().astype(np.float64)

    def obj_func_np(self, rama_):
        rama = torch.DoubleTensor(rama_)
        rama = rama.view(self.L, 21)
        with torch.no_grad():
            f = self.energy(rama) * Scale_factor
            return f.item()

    def foldning(self):
        ilter = self.init_ret
        # 1) get initial quaternions (double precision)
        try:
            init_q = self.init_quat(ilter).double()
        except Exception:
            init_q = self.init_quat_safe(ilter).double()

        # 2) move to target device (GPU if available), enable grad
        param = init_q.to(device).clone().detach().requires_grad_(True)

        # 3) set up PyTorch LBFGS optimizer over `param`
        optimizer = opt.LBFGS(
            [param],
            max_iter=self.config.get("max_iter", 300),
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        # 4) define the "closure" that LBFGS will call to reevaluate loss + gradients
        def closure():
            optimizer.zero_grad()  # clear old grads
            E = self.energy(param.view(self.L, 21)) * Scale_factor  # compute ∂E/∂param
            E.backward()
            return E

        # 5) run LBFGS until convergence (it calls closure repeatedly)
        optimizer.step(closure)

        # 6) write out final PDB
        final_energy = self.energy(param.view(self.L, 21)).item()
        self.outpdb(param, self.saveprefix + ".pdb", energystr=str(final_energy))

    def outpdb(self, rama, savefile, start=0, end=10000, energystr=""):
        # bring baseframes and quaternion data onto CPU to prevent device mismatch
        basex_cpu = self.basex.detach().cpu()
        otherx_cpu = self.otherx.detach().cpu()
        sidex_cpu = self.sidex.detach().cpu()
        shaped_rama = rama.view(self.L, 21).detach().cpu()
        # compute backbone and other coords
        coor_np = a2b.quat2b(basex_cpu, shaped_rama[:, 9:]).detach().cpu().numpy()
        other_np = a2b.quat2b(otherx_cpu, shaped_rama[:, 9:]).detach().cpu().numpy()
        coor = torch.FloatTensor(coor_np)
        # compute side atom coords
        side_coor_NP = (  # noqa: F841
            a2b.quat2b(sidex_cpu, torch.cat([shaped_rama[:, :9], coor[:, -1]], dim=-1))
            .detach()
            .cpu()
            .numpy()
        )

        Atom_name = [" P  ", " C4'", " N1 "]
        Other_Atom_name = [" O5'", " C5'", " C3'", " O3'", " C1'"]
        other_last_name = ["O", "C", "C", "O", "C"]

        side_atoms = [  # noqa: F841
            " N1 ",
            " C2 ",
            " O2 ",
            " N2 ",
            " N3 ",
            " N4 ",
            " C4 ",
            " O4 ",
            " C5 ",
            " C6 ",
            " O6 ",
            " N6 ",
            " N7 ",
            " N8 ",
            " N9 ",
        ]
        side_last_name = [  # noqa: F841
            "N",
            "C",
            "O",
            "N",
            "N",
            "N",
            "C",
            "O",
            "C",
            "C",
            "O",
            "N",
            "N",
            "N",
            "N",
        ]

        base_dict = rigid.base_table()  # noqa: F841
        last_name = ["P", "C", "N"]
        wstr = [f"REMARK {str(energystr)}"]
        templet = "%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s"
        count = 1
        for i in range(self.L):
            if self.seq[i] in ["a", "g", "A", "G"]:
                Atom_name = [" P  ", " C4'", " N9 "]
                # atoms = ['P','C4']

            elif self.seq[i] in ["c", "u", "C", "U"]:
                Atom_name = [" P  ", " C4'", " N1 "]
            for j in range(coor_np.shape[1]):
                outs = (
                    "ATOM  ",
                    count,
                    Atom_name[j],
                    self.seq[i],
                    "A",
                    i + 1,
                    coor_np[i][j][0],
                    coor_np[i][j][1],
                    coor_np[i][j][2],
                    0,
                    0,
                    last_name[j],
                    "",
                )
                if i >= start - 1 and i < end:
                    wstr.append(templet % outs)
                    count += 1

            for j in range(other_np.shape[1]):
                outs = (
                    "ATOM  ",
                    count,
                    Other_Atom_name[j],
                    self.seq[i],
                    "A",
                    i + 1,
                    other_np[i][j][0],
                    other_np[i][j][1],
                    other_np[i][j][2],
                    0,
                    0,
                    other_last_name[j],
                    "",
                )
                if i >= start - 1 and i < end:
                    wstr.append(templet % outs)
                    count += 1

        wstr = "\n".join(wstr)
        wfile = open(savefile, "w")
        wfile.write(wstr)
        wfile.close()

    def outpdb_coor(self, coor_np, savefile, start=0, end=1000, energystr=""):
        Atom_name = [" P  ", " C4'", " N1 "]
        last_name = ["P", "C", "N"]
        wstr = [f"REMARK {str(energystr)}"]
        templet = "%6s%5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s"
        count = 1
        for i in range(self.L):
            if self.seq[i] in ["a", "g", "A", "G"]:
                Atom_name = [" P  ", " C4'", " N9 "]

            elif self.seq[i] in ["c", "u", "C", "U"]:
                Atom_name = [" P  ", " C4'", " N1 "]
            for j in range(coor_np.shape[1]):
                outs = (
                    "ATOM  ",
                    count,
                    Atom_name[j],
                    self.seq[i],
                    "A",
                    i + 1,
                    coor_np[i][j][0],
                    coor_np[i][j][1],
                    coor_np[i][j][2],
                    0,
                    0,
                    last_name[j],
                    "",
                )
                if i >= start - 1 and i < end:
                    wstr.append(templet % outs)
                count += 1

        wstr = "\n".join(wstr)
        wfile = open(savefile, "w")
        wfile.write(wstr)
        wfile.close()

    def init_quat(self, ii):
        x = torch.rand([self.L, 21])
        x[:, 18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii]
        biasq = torch.mean(init_coor, dim=1, keepdim=True)
        q = init_coor - biasq
        m = torch.einsum("bnz,bny->bzy", self.basex, q).reshape([self.L, -1])
        x[:, :9] = x[:, 9:18] = m
        x.requires_grad_()
        return x

    def init_quat_safe(self, ii):
        x = torch.rand([self.L, 21])
        x[:, 18:] = self.txs[ii].mean(dim=1)
        init_coor = self.txs[ii]
        biasq = torch.mean(init_coor, dim=1, keepdim=True)
        q = init_coor - biasq + torch.rand([self.L, 3, 3])
        m = (
            torch.einsum("bnz,bny->bzy", self.basex, q) + torch.eye(3)[None, :, :]
        ).reshape([self.L, -1])
        x[:, :9] = x[:, 9:18] = m
        x.requires_grad_()
        return x


if __name__ == "__main__":

    fastafile = sys.argv[1]
    saveprefix = sys.argv[2]
    retdirs = sys.argv[3]
    ret_score = sys.argv[4]
    foldconfig = sys.argv[5]

    # Check for optional AF3 file
    af3file = None
    if len(sys.argv) > 6:
        af3file = sys.argv[6]
        if os.path.exists(af3file):
            print(f"[DRfold2] Using AlphaFold3 structure: {af3file}")
        else:
            print(f"[DRfold2] Warning: AlphaFold3 file not found: {af3file}")
            af3file = None

    savepare = os.path.dirname(saveprefix)
    if not os.path.isdir(savepare):
        os.makedirs(savepare)

    num_of_models = readconfig(foldconfig)["num_of_models"]

    score_dict = readconfig(ret_score)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1])
    lowest_n_keys = [item[0] for item in sorted_items][:num_of_models]
    bestkey = lowest_n_keys[0] + ""
    print("Before sort:", lowest_n_keys)
    lowest_n_keys.sort()
    print("After sort:", lowest_n_keys)
    bestindex = lowest_n_keys.index(bestkey)

    current_ret = bestkey
    retfiles = [os.path.join(retdirs, afile) for afile in lowest_n_keys]
    stru = Structure(
        fastafile,
        retfiles,
        saveprefix + "_from_" + current_ret,
        bestindex,
        foldconfig,
        af3file,
    )
    stru.foldning()
