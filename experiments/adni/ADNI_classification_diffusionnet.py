import os
from typing import Iterable
from glob import glob

import numpy as np
import scipy.sparse as sparse
from sklearn import model_selection as ms
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
# turn off tqdm
def tqdm(it, *a, **k):
    return it

import potpourri3d as pp3d

import diffusion_net

# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz', 'normal', 'dcm', or 'hks') default: xyz", default = 'xyz')
parser.add_argument("--split_size", type=int, help="how large of a training set per-class default: 10", default=10)
args = parser.parse_args()

# system things
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32

# problem/dataset things
n_class = 2

# model 
input_features = args.input_features # one of ['xyz', 'hks', 'normal', 'dcm']
k_eig = 128

# training settings
n_epoch = 300
lr = 1e-3
decay_every = 50
decay_rate = 0.5
augment_random_rotate = input_features in ['xyz', 'normal']
label_smoothing_fac = 0.2


# Important paths
base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, "../..", "data", "adni")
op_cache_dir = os.path.join(base_path, "../..", "data", "adni_diffusionnet")
dcm_path = os.path.join(dataset_path, "dcm_coords.npz")


class ADNI_Dataset(Dataset):

    def __init__(self,
                 root_dir,
                 k_eig,
                 files: Iterable,
                 y: Iterable,
                 op_cache_dir=None):

        self.root_dir = root_dir
        self.n_class = n_class
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir

        self.class_names = ["CN", "AD"]
        self.entries = {c: set() for c in self.class_names}

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []
        self.dcm_coords = []

        dcm = np.load(dcm_path)

        for f, c in zip(files, y):
            name = os.path.basename(f)[:-4]

            verts, faces = pp3d.read_mesh(f)

            # map dcm coords to vertices
            R, S = dcm[name]
            f2v = sparse.csr_matrix((np.ones(faces.size), (faces.flat, np.repeat(np.arange(len(faces)), 3))))
            deg = np.array(f2v.sum(axis=1)).squeeze()
            Rv = (f2v @ R.reshape(-1, 9)) / deg[:, None]
            Sv = (f2v @ S.reshape(-1, 9)) / deg[:, None]
            self.dcm_coords.append(torch.tensor(np.c_[Rv, Sv]).float())

            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)
            # center only
            #verts = (verts - torch.mean(verts, dim=-2, keepdim=True))

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(c)
            self.entries[self.class_names[c]].add(name)

        for ind, label in enumerate(self.labels_list):
            self.labels_list[ind] = torch.tensor(label)

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        verts = self.verts_list[idx]
        faces = self.faces_list[idx]
        label = self.labels_list[idx]
        dcm = self.dcm_coords[idx]
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, faces,
                                                                                           k_eig=self.k_eig,
                                                                                           op_cache_dir=self.op_cache_dir)
        return verts, faces, frames, mass, L, evals, evecs, gradX, gradY, label, dcm


def train(train_data, val_data, test_data):
    # === Create the model

    C_in = {'normal': 3, 'xyz': 3, 'hks': 16, 'dcm': 18}[input_features]  # dimension of input features

    model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                              C_out=n_class,
                                              C_width=64,
                                              N_block=4,
                                              last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                              outputs_at='global_mean',
                                              dropout=False)

    model = model.to(device)
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # === Optimize
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoch):

        # Implement lr decay
        if epoch > 0 and epoch % decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate


        # Set model to 'train' mode
        model.train()
        optimizer.zero_grad()

        correct = 0
        total_num = 0
        for data in tqdm(train_data):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, dcm = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)

            # Randomly rotate positions
            if augment_random_rotate:
                verts = diffusion_net.utils.random_rotate_points(verts)

            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'normal':
                 features = frames[:,2,:]
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            elif input_features == 'dcm':
                features = dcm.to(device)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # Evaluate loss
            loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
            loss.backward()

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

            # Step the optimizer
            optimizer.step()
            optimizer.zero_grad()

        train_acc = correct / total_num

        val_acc = test(model, val_data)
        test_acc = test(model, test_data)
        print("Epoch {} - Train: {:06.3f}%  Val: {:06.3f}%  Test: {:06.3f}%".format(epoch, 100 * train_acc,
                                                                                    100 * val_acc, 100 * test_acc))


# Do an evaluation pass on the test dataset 
def test(model, data):
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for d in tqdm(data):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels, dcm = d

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'normal':
                features = frames[:,2,:]
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
            elif input_features == 'dcm':
                features = dcm.to(device)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

    test_acc = correct / total_num
    return test_acc


def read_log(f:str) -> np.ndarray:
    txt = np.loadtxt(f, comments=['Training', 'cache'], dtype=float, usecols=[4, 6, 8],
                     converters=lambda v: float(v[:-1]))
    acc = np.resize(txt, (len(txt) // n_epoch, n_epoch, 3))
    i = np.argmax(acc[..., 1], axis=1)
    return np.array([acc[j, k, 2] for j, k in enumerate(i)])


if __name__ == '__main__':

    # find hippocampi
    files = np.array(sorted(glob('../../data/adni/**/*obj', recursive=True)), dtype=str)
    y = np.array([int(f.split('/')[-2] == 'AD') for f in files])
    assert len(np.unique(y)) == 2


    # monte carlo cross validation
    for seed in range(100):

        # split data in training, validation and testing sets
        idx_train, idx_test = ms.train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        # hold back validation set from the training data
        idx_train, idx_val = ms.train_test_split(idx_train, test_size=0.25, random_state=seed, stratify=y[idx_train])

        # === Load datasets

        # Train dataset
        train_dataset = ADNI_Dataset(dataset_path, k_eig=k_eig, op_cache_dir=op_cache_dir, files=files[idx_train], y=y[idx_train])
        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
        # Validation dataset
        val_dataset = ADNI_Dataset(dataset_path, k_eig=k_eig, op_cache_dir=op_cache_dir, files=files[idx_val], y=y[idx_val])
        val_loader = DataLoader(val_dataset, batch_size=None)
        # Test dataset
        test_dataset = ADNI_Dataset(dataset_path, k_eig=k_eig, op_cache_dir=op_cache_dir, files=files[idx_test], y=y[idx_test])
        test_loader = DataLoader(test_dataset, batch_size=None)

        # run training
        print("Training...")
        train(train_loader, val_loader, test_loader)
