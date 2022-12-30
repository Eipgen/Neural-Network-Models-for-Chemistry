# Neural-Network-Models-for-chemistry

A collection of Nerual Network Models for potential building

## Density Functional Theory Method

- [DeePKS, DeePHF](https://github.com/deepmodeling/deepks-kit)  
DeePKS-kit is a program to generate accurate energy functionals for quantum chemistry systems, for both perturbative scheme (DeePHF) and self-consistent scheme (DeePKS).

- [NeuralXC](https://github.com/semodi/neuralxc) Implementation of a machine learned density functional.

- [MOB-ML](https://aip.scitation.org/doi/10.1063/5.0032362) 
Machine Learning for Molecular Orbital Theory.

- [DM21](https://github.com/deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)  
Pushing the Frontiers of Density Functionals by Solving the Fractional Electron Problem.
- [NN-GGA, NN-NRA, NN-meta-GGA, NN-LSDA](https://github.com/ml-electron-project/NNfunctional)  
Completing density functional theory by machine-learning hidden messages from molecules.
- [FemiNet](https://github.com/deepmind/ferminet)  
FermiNet is a neural network for learning highly accurate ground state wavefunctions of atoms and molecules using a variational Monte Carlo approach.
- [DeePQMC](https://github.com/deepqmc/deepqmc)  
DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in PyTorch as trial wave functions.
- [PauliNet](https://www.nature.com/articles/s41557-020-0544-y#Bib1)  
PauliNet builds upon HF or CASSCF orbitals as a physically meaningful baseline and takes a neural network approach to the SJB wavefunction in order tocorrect this baseline towards a high-accuracy solution.
- [DeePErwin](https://github.com/mdsunivie/deeperwin)  
DeepErwin is python package that implements and optimizes wave function models for numerical solutions to the multi-electron Schrödinger equation.
- [Jax-DFT](https://github.com/google-research/google-research/tree/master/jax_dft) 
<br>JAX-DFT implements one-dimensional density functional theory (DFT) in JAX. It uses powerful JAX primitives to enable JIT compilation, automatical differentiation, and high-performance computation on GPUs.

## Molecular Force Field Method

- [DeePMD](https://github.com/deepmodeling/deepmd-kit)
<br>A package designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics.
- [Torch-ANI](https://github.com/aiqm/torchani)
<br>TorchANI is a pytorch implementation of ANI model.
- [Schrodinger-ANI](http://public-sani.onschrodinger.com/)  
A neural network potential energy function for use in drug discovery, with chemical element support extended from 41% to 94% of druglike molecules based on ChEMBL.
- [NerualForceFild](https://github.com/learningmatter-mit/NeuralForceField)
<br>The Neural Force Field (NFF) code is an API based on SchNet, DimeNet, PaiNN and DANN . It provides an interface to train and evaluate neural networks for force fields. It can also be used as a property predictor that uses both 3D geometries and 2D graph information.
- [NNPOps](https://github.com/openmm/NNPOps)
<br>The goal of this project is to promote the use of neural network potentials (NNPs) by providing highly optimized, open source implementations of bottleneck operations that appear in popular potentials.
- [Nequip](https://github.com/mir-group/nequip)
<br>NequIP is an open-source code for building E(3)-equivariant interatomic potentials.
- [E3NN](https://github.com/e3nn/e3nn)  
Euclidean neural networks,The aim of this library is to help the development of E(3) equivariant neural networks. It contains fundamental mathematical operations such as tensor products and spherical harmonics.
- [SchNet](https://github.com/atomistic-machine-learning/SchNet)
<br>SchNet is a deep learning architecture that allows for spatially and chemically resolved insights into quantum-mechanical observables of atomistic systems.
- [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
<br>SchNetPack aims to provide accessible atomistic neural networks that can be trained and applied out-of-the-box, while still being extensible to custom atomistic architectures.
- [G-SchNet](https://github.com/atomistic-machine-learning/G-SchNet)
<br>Implementation of G-SchNet - a generative model for 3d molecular structures.
- [PhysNet](https://github.com/MMunibas/PhysNet)
<br>PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges.
- [DimeNet](https://github.com/gasteigerjo/dimenet)
<br>Directional Message Passing Neural Network.
- [GemNet](https://github.com/TUM-DAML/gemnet_pytorch)
<br>Universal Directional Graph Neural Networks for Molecules.
- [DeePMoleNet](https://github.com/Frank-LIU-520/DeepMoleNet)
<br>DeepMoleNet is a deep learning package for molecular properties prediction.
- [AirNet](https://github.com/helloyesterday/AirNet)
<br>A new GNN-based deep molecular model by MindSpore.
- [TorchMD-Net](https://github.com/torchmd/torchmd-net)
<br>TorchMD-NET provides graph neural networks and equivariant transformer neural networks potentials for learning molecular potentials.
- [AQML](https://github.com/binghuang2018/aqml)
<br>AQML is a mixed Python/Fortran/C++ package, intends to simulate quantum chemistry problems through the use of the fundamental building blocks of larger systems.
- [TensorMol](https://github.com/jparkhill/TensorMol)
<br>A pakcages of NN model chemistry, contains Behler-Parrinello with electrostatics, Many Body Expansion Bonds in Molecules NN, Atomwise, Forces, Inductive Charges.
- [SpookyNet](https://github.com/OUnke/SpookyNet)
<br>Spookynet: Learning force fields with electronic degrees of freedom and nonlocal effects.
- [RuNNer](https://www.uni-goettingen.de/de/software/616512.html)
<br>A program package for constructing high-dimensional neural network potentials,4G-HDNNPs,3G-HDNNPs.
- [aenet](https://github.com/atomisticnet/aenet)
<br>The Atomic Energy NETwork (ænet) package is a collection of tools for the construction and application of atomic interaction potentials based on artificial neural networks.
- [sGDML](http://www.sgdml.org/)
<br> Symmetric Gradient Domain Machine Learning
- [GAP](https://github.com/libAtoms/GAP)
<br>This package is part of QUantum mechanics and Interatomic Potentials
- [QUIP](https://github.com/libAtoms/QUIP)
<br>The QUIP package is a collection of software tools to carry out molecular dynamics simulations. It implements a variety of interatomic potentials and tight binding quantum mechanics, and is also able to call external packages, and serve as plugins to other software such as LAMMPS, CP2K and also the python framework ASE.
- [NNP-MM](https://github.com/RowleyGroup/NNP-MM)
<br>NNP/MM embeds a Neural Network Potential into a conventional molecular mechanical (MM) model.
- [GAMD](https://github.com/BaratiLab/GAMD)
<br>Data and code for Graph neural network Accelerated Molecular Dynamics.
- [PFP](https://matlantis.com/)
<br>Here we report a development of universal NNP called PreFerred Potential (PFP), which is able to handle any combination of 45 elements.
Particular emphasis is placed on the datasets, which include a diverse set of virtual structures used to attain the universality.
- [TeaNet](https://codeocean.com/capsule/4358608/tree)
<br>universal neural network interatomic potential inspired by iterative electronic relaxations.
- [n2p2](https://github.com/CompPhysVienna/n2p2)
<br>This repository provides ready-to-use software for high-dimensional neural network potentials in computational physics and chemistry.
- [AIMNET](https://github.com/aiqm/aimnet)  
This repository contains reference AIMNet implementation along with some examples and menchmarks.
- [charge_transfer_nnp](https://github.com/pfnet-research/charge_transfer_nnp)  
About Graph neural network potential with charge transfer with neuqip model.
- [AMP](https://amp.readthedocs.io/en/latest/)  
Amp: A modular approach to machine learning in atomistic simulations(<https://github.com/ulissigroup/amptorch>)
- [SCFNN](https://github.com/andy90/SCFNN)  
A self consistent field neural network (SCFNN) model.
- [jax-md](https://github.com/google/jax-md)  
 JAX MD is a functional and data driven library. Data is stored in arrays or tuples of arrays and functions transform data from one state to another.
- [EANN](https://github.com/zhangylch/EANN)  
Embedded Atomic Neural Network (EANN) is a physically-inspired neural network framework. The EANN package is implemented using the PyTorch framework used to train interatomic potentials, dipole moments, transition dipole moments and polarizabilities of various systems.
- [espaloma](https://github.com/choderalab/espaloma)  
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm.
- [MDsim](https://github.com/kyonofx/MDsim)  
Training and simulating MD with ML force fields
- [ForceNet](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/forcenet.py) 
We demonstrate that force-centric GNN models without any explicit physical constraints are able to predict atomic forces more accurately than state-of-the-art energy centric GNN models, while being faster both in training
and inference.
- [DIG](https://github.com/divelab/DIG)  
A library for graph deep learning research.
- [scn](https://github.com/Open-Catalyst-Project/ocp/tree/main/ocpmodels/models/scn)  
Spherical Channels for Modeling Atomic Interactions
- [spinconv](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py)  
Rotation Invariant Graph Neural Networks using Spin Convolutions.
- [HIPPYNN](https://github.com/lanl/hippynn)  
a modular library for atomistic machine learning with pytorch.
- [VisNet](https://github.com/microsoft/ViSNet)     
a scalable and accurate geometric deep learning potential for molecular dynamics simulation

## Semi-Empirical Method

- [OrbNet](https://arxiv.org/abs/2107.00299)
<br>OrbNet Denali: A machine learning potential for biological and organic chemistry with semi-empirical cost and DFT accuracy.

- [AIQM1](https://doi.org/10.1038/s41467-021-27340-2)
<br>Artificial intelligence-enhanced quantum chemical method with broad applicability.

- [BpopNN](https://doi.org/10.1021/acs.jctc.0c00217)
<br>Incorporating Electronic Information into Machine Learning Potential Energy Surfaces via Approaching the Ground-State Electronic Energy as a Function of Atom-Based Electronic Populations.

- [Delfta](https://github.com/josejimenezluna/delfta)
<br>The DelFTa application is an easy-to-use, open-source toolbox for predicting quantum-mechanical properties of drug-like molecules. Using either ∆-learning (with a GFN2-xTB baseline) or direct-learning (without a baseline), the application accurately approximates DFT reference values (ωB97X-D/def2-SVP).

## Coarse-Grained Method

- [cgnet](https://github.com/coarse-graining/cgnet)
<br>Coarse graining for molecular dymamics
- [SchNet-CG](https://arxiv.org/ftp/arxiv/papers/2209/2209.12948.pdf)  
We explore the application of SchNet models to obtain a CG potential for liquid benzene, investigating the effect of model architecture and hyperparameters on the thermodynamic, dynamical, and structural properties of the simulated CG systems, reporting and discussing challenges encountered and future directions envisioned.

## Enhanced Sampling Method

- [VES-NN](https://doi.org/10.1073/pnas.1907975116)
<br>[Neural networks-based variationallyenhanced sampling]

- [Accelerated_sampling_with_autoencoder](https://github.com/weiHelloWorld/accelerated_sampling_with_autoencoder)
<br>[Accelerated sampling framework with autoencoder-based method]

## QM/MM Model

- [NNP-MM](https://github.com/RowleyGroup/NNP-MM)
<br>NNP/MM embeds a Neural Network Potential into a conventional molecular mechanical (MM) model. We have implemented this using the Custom QM/MM features of NAMD 2.13, which interface NAMD with the TorchANI NNP python library developed by the Roitberg and Isayev groups.
- [DeeP-HP](https://github.com/TinkerTools/tinker-hp/tree/Deep-HP)
<br>support for neural networks potentials (ANI-2X, DeepMD).

## Charge Model

- [gimlet](https://github.com/choderalab/gimlet)  
Graph Inference on MoLEcular Topology. A package for modelling, learning, and inference on molecular topological space written in Python and TensorFlow.

## Post-HF Method
- [LNO-CCSD(T)](10.1021/acs.jctc.8b00442)
- [DLPNO-CCSD(T)](https://doi.org/10.1021/acs.jpca.0c11270)
- [PNO-CCSD(T)](https://doi.org/10.1021/acs.jctc.7b00180)
