"""
A MIXIN is a programming concept used in object-oriented languages like 
Python to promote code reuse and maintainability.  A mixin is essentially a 
class that provides a certain functionality or behavior that can be easily 
added to other classes by inheriting from it.

Here are some key points about mixins:

1. Reusability: 
    Mixins encapsulate specific functionality that can be reused across multiple 
    classes without the need for code duplication. This promotes a modular 
    approach to programming.

2. Composition over Inheritance: 
    Mixins allow you to add functionality to classes without the need for deep 
    inheritance hierarchies. Instead of creating complex inheritance structures,
    you can compose functionality by combining mixins with regular classes.

3. Single Responsibility Principle: 
    Mixins often follow the Single Responsibility Principle (SRP), which states 
    that a class should have only one reason to change. By isolating specific 
    functionalities in mixins, each class can focus on its primary responsibility.

4. Flexibility: 
    Mixins can be easily combined with other mixins and classes to create new 
    classes with customized behavior. This allows for greater flexibility 
    in designing class hierarchies.

Mixins are typically implemented as classes that are not meant to be instantiated 
on their own but are intended to be inherited by other classes. They provide 
additional methods or attributes that enhance the functionality of the subclass. 
By inheriting from multiple mixins, a subclass can combine various functionalities 
from different sources.
"""

import inspect
import os
import pickle
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from torch.distributions import Normal
from anndata import AnnData, read
from scipy.sparse import issparse

from ._utils import UnpicklerCpu, _validate_var_names

# Adapted from
# Title: Biologically informed deep learning to query gene programs in single-cell atlases
# Authors: Mohammad Lotfollahi, Sergei Rybakov, Karin Hrovatin, Soroor Hediyeh-zadeh, Carlos Talavera-López, Alexander V. Misharin & Fabian J. Theis 
# Code: https://github.com/theislab/scarches/tree/master/scarches/models/base/_base.py

training_params_to_keep = [
    'batch_size', 
    'alpha_kl', 
    'alpha_kl_epoch_anneal', 
    'alpha_kl_iter_anneal', 
    'use_early_stopping', 
    'early_stopping', 
    'reload_best', 
    'n_samples', 
    'train_frac', 
    'use_stratified_sampling', 
    'weight_decay', 
    'clip_value', 
    'n_workers', 
    'seed', 
    'monitor', 
    'monitor_only_val', 
    'device', 
    'epoch', 
    'n_epochs', 
    'iter', 
    'best_epoch', 
    # 'best_state_dict', 
    'optimizer', 
    'training_time', 
    'iters_per_epoch', 
    'val_iters_per_epoch', 
    'logs', 
    'print_stats', 
    'alpha_gl', 
    'omega', 
    'gamma_ext', 
    'gamma_epoch_anneal', 
    'gamma_anneal_each', 
    'alpha_l1', 
    'alpha_l1_epoch_anneal', 
    'alpha_l1_anneal_each', 
    'beta', 
    'watch_lr', 
    'use_prox_ops', 
    'prox_ops', 
    'corr_coeffs', 
    'initial_lr', 
    'iter_logs']

class BaseMixin:
    """ Adapted from
        Title: scvi-tools
        Authors: Romain Lopez <romain_lopez@gmail.com>,
                 Adam Gayoso <adamgayoso@berkeley.edu>,
                 Galen Xing <gx2113@columbia.edu>
        Date: 14.12.2020
        Code version: 0.8.0-beta.0
        Availability: https://github.com/YosefLab/scvi-tools
        Link to the used code:
        https://github.com/YosefLab/scvi-tools/blob/0.8.0-beta.0/scvi/core/models/base.py
    """
    def _get_user_attributes(self):
        # returns all the self attributes defined in a model class, eg, self.is_trained_
        # attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        attributes = [(name, value) for name, value in self.__dict__.items()]#vars(self).items()]# if not name.startswith("__") and not name.startswith("_abc_")]
        attributes = [
            a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        attributes = [a for a in attributes if not a[0].startswith("_abc_")]
        return attributes

    def _get_public_attributes(self):
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if a[0][-1] == "_"}
        return public_attributes
    
    def _get_training_params(self):
        training_params = [(name, value) for name, value in self.trainer.__dict__.items()]
        training_params = [a for a in training_params]
        training_params = {a[0]: a[1] for a in training_params if a[0] in training_params_to_keep}
        return training_params

    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        anndata_write_kwargs=None, # 'added = None'
    ):
        """Save the state of the model.
           Neither the trainer optimizer state nor the trainer history are saved.
           Parameters
           ----------
           dir_path
                Path to a directory.
           overwrite
                Overwrite existing data or not. If `False` and directory
                already exists at `dir_path`, error will be raised.
           save_anndata
                If True, also saves the anndata
           anndata_write_kwargs
                Kwargs for anndata write function
        """
        # get all the public attributes
        public_attributes = self._get_public_attributes()
        training_params = self._get_training_params()

        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )

        if save_anndata:
            self.adata.write(
                os.path.join(dir_path, "adata.h5ad"), anndata_write_kwargs
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        train_params_save_path = os.path.join(dir_path, "train_params.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")
        
        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")

        torch.save(self.model.state_dict(), model_save_path)

        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)

        with open(train_params_save_path, "wb") as f:
            pickle.dump(training_params, f)

    def _load_expand_params_from_dict(self, state_dict):
        load_state_dict = state_dict.copy()

        device = next(self.model.parameters()).device

        new_state_dict = self.model.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                load_ten = load_ten.to(device)
                # only one dim diff
                new_shape = new_ten.shape
                n_dims = len(new_shape)
                sel = [slice(None)] * n_dims
                for i in range(n_dims):
                    dim_diff = new_shape[i] - load_ten.shape[i]
                    axs = i
                    sel[i] = slice(-dim_diff, None)
                    if dim_diff > 0:
                        break
                fixed_ten = torch.cat([load_ten, new_ten[tuple(sel)]], dim=axs)
                load_state_dict[key] = fixed_ten

        for key, ten in new_state_dict.items():
            if key not in load_state_dict:
                load_state_dict[key] = ten

        self.model.load_state_dict(load_state_dict)

    @classmethod
    def _load_params(cls, dir_path: str, map_location: Optional[str] = None):
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        training_dict_path = os.path.join(dir_path, "train_params.pkl")

        try:
            with open(setup_dict_path, "rb") as handle:
                attr_dict = pickle.load(handle)
            with open(training_dict_path, "rb") as handle:
                train_dict = pickle.load(handle)
        # This catches the following error:
        # RuntimeError: Attempting to deserialize object on a CUDA device
        # but torch.cuda.is_available() is False.
        # If you are running on a CPU-only machine, please use torch.load with
        # map_location=torch.device('cpu') to map your storages to the CPU.
        except RuntimeError:
            with open(setup_dict_path, "rb") as handle:
                attr_dict = UnpicklerCpu(handle).load()
            with open(training_dict_path, "rb") as handle:
                train_dict = UnpicklerCpu(handle).load()

        model_path = os.path.join(dir_path, "model_params.pt")
        model_state_dict = torch.load(model_path, map_location=map_location)

        varnames_path = os.path.join(dir_path, "var_names.csv")
        var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

        return attr_dict, train_dict, model_state_dict, var_names

    @classmethod
    def create_model_directory(cls, dataset,
                                lipids_format,
                                mask_key,
                                soft_mask,
                                condition_key,
                                hidden_layer_sizes,
                                use_bn,
                                
                                percentage, 
                                n_epochs,
                                batch_size,
                                dr_rate,
                                initial_lr,
                                weight_decay,
                                alpha_kl,
                                alpha_gl,
                                alpha_l1,
                                initialization=None,
                                clipping_decoder_weights=False,):
        """
        Create a loading directory based on the architecture and training parameters.
        """
        arch_path = os.path.join(f"{lipids_format}_lipids", 
                                f"mask_{mask_key}", 
                                'soft' if soft_mask else 'hard', 
                                condition_key if condition_key is not None else 'no',
                                f"encoder_l_{'_'.join(map(str, hidden_layer_sizes))}", 
                                'batch_norm' if use_bn else 'layer_norm')

        train_path = f"{dataset}_{round(percentage*100)}percent_epochs{n_epochs}_" \
                        f"bs{batch_size}_dr{dr_rate}_" \
                        f"lr{initial_lr}_wd{weight_decay}_" \
                        f"kl{alpha_kl}_gl{alpha_gl}_l1{alpha_l1}" \
                        f"_init{initialization}_clip{clipping_decoder_weights}"

        full_path = os.path.join(arch_path, train_path)

        return full_path

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        map_location = None
    ):
        """Instantiate a model from the saved output.
           Parameters
           ----------
           dir_path
                Path to saved outputs.
           adata
                AnnData object.
                If None, will check for and load anndata saved with the model.
           map_location
                 a function, torch.device, string or a dict specifying
                 how to remap storage locations
           Returns
           -------
                Model with loaded state dictionaries.
        """
        adata_path = os.path.join(dir_path, "adata.h5ad")

        load_adata = adata is None

        if os.path.exists(adata_path) and load_adata:
            adata = read(adata_path)
        elif not os.path.exists(adata_path) and load_adata:
            raise ValueError("Save path contains no saved anndata and no adata was passed.")

        attr_dict, train_dict, model_state_dict, var_names = cls._load_params(dir_path, map_location)
        
        # Overwrite adata with new lipids
        adata = _validate_var_names(adata, var_names)
        
        cls._validate_adata(adata, attr_dict)
        init_params = cls._get_init_params_from_dict(attr_dict)
        
        model = cls(adata, **init_params)
        model.model.device = next(iter(model_state_dict.values())).device
        model.model.to(model.model.device)
        model.model.load_state_dict(model_state_dict)
        model.model.eval()

        model.is_trained_ = attr_dict['is_trained_']
        model.training_params = train_dict
        
        return model


class SurgeryMixin:
    @classmethod
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, 'Model'],
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
        map_location = None,
        kwargs = None, # added '= None'
    ):
        """Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

           Parameters
           ----------
           adata
                Query anndata object.
           reference_model
                A model to expand or a path to a model folder.
           freeze: Boolean
                If 'True' freezes every part of the network except the first layers of encoder/decoder.
           freeze_expression: Boolean
                If 'True' freeze every weight in first layers except the condition weights.
           remove_dropout: Boolean
                If 'True' remove Dropout for Transfer Learning.
           map_location
                map_location to remap storage locations (as in '.load') of 'reference_model'.
                Only taken into account if 'reference_model' is a path to a model on disk.
           kwargs
                kwargs for the initialization of the query model.

           Returns
           -------
           new_model
                New model to train on query data.
        """
        # NOT SAVING TRAIN_DICT IN THE MODEL FOR THE ALREADY TRAINED MODEL

        if isinstance(reference_model, str):
            attr_dict, train_dict, model_state_dict, var_names = cls._load_params(reference_model, map_location)
            adata = _validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            train_dict = reference_model._get_training_params()
            model_state_dict = reference_model.model.state_dict()
            adata = _validate_var_names(adata, reference_model.adata.var_names)
        
        init_params = deepcopy(cls._get_init_params_from_dict(attr_dict))

        conditions = init_params['conditions']
        condition_key = init_params['condition_key']
        
        if condition_key is not None:    
            new_conditions = []
            adata_conditions = adata.obs[condition_key].unique().tolist()
            
            # Check if new conditions are already known
            for item in adata_conditions:
                if item not in conditions:
                    new_conditions.append(item)

            # Add new conditions to overall conditions
            for condition in new_conditions:
                conditions.append(condition)
        
        if remove_dropout:
            init_params['dr_rate'] = 0.0
        
        if kwargs is not None:
            init_params.update(kwargs)
        
        new_model = cls(adata, **init_params)
        new_model.model.to(next(iter(model_state_dict.values())).device)
        new_model._load_expand_params_from_dict(model_state_dict)
        
        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'theta' in name:
                    p.requires_grad = True
                if freeze_expression:
                    if 'cond_L.weight' in name:
                        p.requires_grad = True
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = True

        return new_model


class CVAELatentsMixin:
    def get_latent(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        mean: bool = False,
        mean_var: bool = False
    ):
        """Map `x` in to the latent space. This function will feed data in encoder and return z for each sample in
           data.
           Parameters
           ----------
           x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
           c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
           mean
                return mean instead of random sample from the latent space
           mean_var
                return mean and variance instead of random sample from the latent space
                if `mean=False`.
           Returns
           -------
                Returns array containing latent space encoding of 'x'.
        """
        device = next(self.model.parameters()).device
        if x is None and c is None:
            x = self.adata.X
            if self.conditions_ is not None and self.conditions_ != []:
                c = self.adata.obs[self.condition_key_]

        if c is not None:
            c = np.asarray(c)
            if not set(c).issubset(self.conditions_):
                raise ValueError("Incorrect conditions")
            labels = np.zeros(c.shape[0])
            for condition, label in self.model.condition_encoder.items():
                labels[c == condition] = label
            c = torch.tensor(labels, device=device)

        latents = []
        indices = torch.arange(x.shape[0])
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            x_batch = x[batch, :]
            if issparse(x_batch):
                x_batch = x_batch.toarray()
            x_batch = torch.tensor(x_batch, device=device)

            print(f"x_batch: {x_batch.shape}, c[batch]: {c[batch].shape}")
            print(f"mean: {mean}, mean_var: {mean_var}")
            
            latent = self.model.get_latent(x_batch, c[batch] if c is not None else None, mean, mean_var)
            latent = (latent,) if not isinstance(latent, tuple) else latent
            latents += [tuple(l.cpu().detach() for l in latent)]

        result = tuple(np.array(torch.cat(l)) for l in zip(*latents))
        result = result[0] if len(result) == 1 else result

        return result

    def get_y(
        self,
        x: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
    ):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.

           Parameters
           ----------
           x
                Numpy nd-array to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
                If None, then `self.adata.X` is used.
           c
                `numpy nd-array` of original (unencoded) desired labels for each sample.
           Returns
           -------
                Returns array containing output of first decoder layer.
        """
        device = next(self.model.parameters()).device
        if x is None and c is None:
            x = self.adata.X
            if self.conditions_ is not None:
                c = self.adata.obs[self.condition_key_]

        if c is not None:
            c = np.asarray(c)
            if not set(c).issubset(self.conditions_):
                raise ValueError("Incorrect conditions")
            labels = np.zeros(c.shape[0])
            for condition, label in self.model.condition_encoder.items():
                labels[c == condition] = label
            c = torch.tensor(labels, device=device)

        latents = []
        indices = torch.arange(x.shape[0])
        subsampled_indices = indices.split(512)
        for batch in subsampled_indices:
            x_batch = x[batch, :]
            if issparse(x_batch):
                x_batch = x_batch.toarray()
            x_batch = torch.tensor(x_batch, device=device)
            latent = self.model.get_y(x_batch, c[batch] if c is not None else None)
            latents += [latent.cpu().detach()]
            
        return np.array(torch.cat(latents))


class CVAELatentsModelMixin:
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.

           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.

           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x, c=None, mean=False, mean_var=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return z for each sample in
           data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           mean: boolean
           Returns
           -------
           Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        
        print(f"x_: {x_.shape}, c: {c.shape}")
        z_mean, z_log_var = self.encoder(x_, c)
        
        latent = self.sampling(z_mean, z_log_var)
        
        if mean:
            return z_mean
        elif mean_var:
            return (z_mean, torch.exp(z_log_var) + 1e-4)
        return latent

    def get_y(self, x, c=None):
        """Map `x` in to the y dimension (First Layer of Decoder). This function will feed data in encoder  and return
           y for each sample in data.

           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.

           Returns
           -------
           Returns Torch Tensor containing output of first decoder layer.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        output = self.decoder(latent, c)
        return output[0]
