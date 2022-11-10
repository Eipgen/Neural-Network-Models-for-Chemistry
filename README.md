# Neural-Network-Models-for-chemistry
A collection of Nerual Network Models for potential building

# Functional

- DeePKS, DeePHF 
<br>[Deepmodeling/deepks-kit: a package for developing machine learning-based chemically accurate energy and density functional models](https://github.com/deepmodeling/deepks-kit)

- NeuralXC 
<br>[Implementation of a machine learned density functional](https://github.com/semodi/neuralxc)

- MOB-ML
<br>[Machine Learning for Molecular Orbital Theory](https://aip.scitation.org/doi/10.1063/5.0032362)

- DM21
<br>[Pushing the Frontiers of Density Functionals by Solving the Fractional Electron Problem](https://github.com/deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)
- NN-GGA,NN-NRA,NN-meta-GGA,NN-LSDA
<br>[Completing density functional theory by machine-learning hidden messages from molecules](https://github.com/ml-electron-project/NNfunctional)
- FemiNet(https://github.com/deepmind/ferminet)
<br>FermiNet is a neural network for learning highly accurate ground state wavefunctions of atoms and molecules using a variational Monte Carlo approach.
- deepqmc(https://github.com/deepqmc/deepqmc)
<br>DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in PyTorch as trial wave functions. Besides the core functionality, it contains implementations of the following ansatzes:
- PauliNet(https://www.nature.com/articles/s41557-020-0544-y#Bib1)
<br>PauliNet builds upon HF or CASSCF orbitals as a physically meaningful baseline and takes a neural network approach to the SJB wavefunction in order tocorrect this baseline towards a high-accuracy solution
- DeePErwin(https://github.com/mdsunivie/deeperwin)
<br>DeepErwin is python package that implements and optimizes wave function models for numerical solutions to the multi-electron Schrödinger equatio


# Molecular Field

- [DeePMD](https://github.com/deepmodeling/deepmd-kit) 
<br>A package designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD)
- [Torch-ANI](https://github.com/aiqm/torchani)
<br>TorchANI is a pytorch implementation of ANI
- [NerualForceFild](https://github.com/learningmatter-mit/NeuralForceField)
<br>The Neural Force Field (NFF) code is an API based on SchNet, DimeNet, PaiNN and DANN . It provides an interface to train and evaluate neural networks for force fields. It can also be used as a property predictor that uses both 3D geometries and 2D graph information
- [NNPOps](https://github.com/openmm/NNPOps)
<br>The goal of this project is to promote the use of neural network potentials (NNPs) by providing highly optimized, open source implementations of bottleneck operations that appear in popular potentials. These are the core design principles.
- [Neupiq](https://github.com/mir-group/nequip)
<br>NequIP is an open-source code for building E(3)-equivariant interatomic potentials.
- [SchNet](https://github.com/atomistic-machine-learning/SchNet)
<br>SchNet is a deep learning architecture that allows for spatially and chemically resolved insights into quantum-mechanical observables of atomistic systems.
- [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
<br>SchNetPack aims to provide accessible atomistic neural networks that can be trained and applied out-of-the-box, while still being extensible to custom atomistic architectures.
- [G-SchNet](https://github.com/atomistic-machine-learning/G-SchNet)
<br>Implementation of G-SchNet - a generative model for 3d molecular structures
- [PhysNet](https://github.com/MMunibas/PhysNet)
<br>PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges
- [DimeNet](https://github.com/gasteigerjo/dimenet)
<br>Directional Message Passing Neural Network
- [GemNet](https://github.com/TUM-DAML/gemnet_pytorch)
<br>Universal Directional Graph Neural Networks for Molecules
- [DeePMoleNet](https://github.com/Frank-LIU-520/DeepMoleNet)
<br>DeepMoleNet is a deep learning package for molecular properties prediction
- [AirNet](https://github.com/helloyesterday/AirNet)
<br>A new GNN-based deep molecular model by Mindspore
- [TorchMD-Net](https://github.com/torchmd/torchmd-net)
<br>TorchMD-NET provides graph neural networks and equivariant transformer neural networks potentials for learning molecular potentials
- [AQML](https://github.com/binghuang2018/aqml)
<br>AQML is a mixed Python/Fortran/C++ package, intends to simulate quantum chemistry problems through the use of the fundamental building blocks of larger systems
- [TensorMol](https://github.com/jparkhill/TensorMol)
<br>pakcages of NN model chemistry
- [SpookyNet](https://github.com/OUnke/SpookyNet)
<br>Spookynet: Learning force fields with electronic degrees of freedom and nonlocal effects
- [RuNNer](https://www.uni-goettingen.de/de/software/616512.html)
<br>a program package for constructing high-dimensional neural network potentials,4G-HDNNPs,3G-HDNNPs
- [aenet](https://github.com/atomisticnet/aenet)
<br>The Atomic Energy NETwork (ænet) package is a collection of tools for the construction and application of atomic interaction potentials based on artificial neural networks
- [sGDML](http://www.sgdml.org/)
<br> Symmetric Gradient Domain Machine Learning
- [GAP](https://github.com/libAtoms/GAP)
<br>This package is part of QUantum mechanics and Interatomic Potentials
- [QUIP](https://github.com/libAtoms/QUIP)
<br>The QUIP package is a collection of software tools to carry out molecular dynamics simulations. It implements a variety of interatomic potentials and tight binding quantum mechanics, and is also able to call external packages, and serve as plugins to other software such as LAMMPS, CP2K and also the python framework ASE.
- [NNP-MM](https://github.com/RowleyGroup/NNP-MM)
<br>NNP/MM embeds a Neural Network Potential into a conventional molecular mechanical (MM) model
- [GAMD](https://github.com/BaratiLab/GAMD)
<br>Data and code for Graph neural network Accelerated Molecular Dynamics
- [PFP](https://matlantis.com/)
<br>Here we report a development of universal NNP called PreFerred Potential (PFP), which is able to handle any combination of 45 elements.
Particular emphasis is placed on the datasets, which include a diverse set of virtual structures used to attain the universality.
- [TeaNet](https://codeocean.com/capsule/4358608/tree)
<br>universal neural network interatomic potential inspired by iterative electronic relaxations
- [n2p2](https://github.com/CompPhysVienna/n2p2)
<br>This repository provides ready-to-use software for high-dimensional neural network potentials in computational physics and chemistry.

- [aimnet](https://github.com/aiqm/aimnet)
This repository contains reference AIMNet implementation along with some examples and menchmarks
- [charge_transfer_nnp](https://github.com/pfnet-research/charge_transfer_nnp)
About Graph neural network potential with charge transfer
- [AMP,AMPTorch](https://amp.readthedocs.io/en/latest/)
Amp: A modular approach to machine learning in atomistic simulations(https://github.com/ulissigroup/amptorch)
- [SCFNN](https://github.com/andy90/SCFNN)
self consistent field neural network (SCFNN) model.
# Semi-Empirical Method


- OrbNet
<br>[OrbNet Denali: A machine learning potential for biological and organic chemistry with semi-empirical cost and DFT accuracy](https://arxiv.org/abs/2107.00299)

- AIQM1
<br>[Artificial intelligence-enhanced quantum chemical method with broad applicability](https://www.nature.com/articles/s41467-021-27340-2)
- BpopNN 
<br>[Incorporating Electronic Information into Machine Learning Potential Energy Surfaces via Approaching the Ground-State Electronic Energy as a Function of Atom-Based Electronic Populations](https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.0c00217)

- Delfta(https://github.com/josejimenezluna/delfta)
<br>The DelFTa application is an easy-to-use, open-source toolbox for predicting quantum-mechanical properties of drug-like molecules. Using either ∆-learning (with a GFN2-xTB baseline) or direct-learning (without a baseline), the application accurately approximates DFT reference values (ωB97X-D/def2-SVP). 
# Coarse-Grained 
- cgnet
(https://github.com/coarse-graining/cgnet)
<br>Coarse graining for molecular dymamics
- a review form arxiv(https://arxiv.org/ftp/arxiv/papers/2209/2209.12948.pdf)
# Enhanaced Sampling
- VES-NN
<br>[Neural networks-based variationallyenhanced sampling](https://www.pnas.org/doi/epdf/10.1073/pnas.1907975116)

- Accelerated_sampling_with_autoencoder
<br>[Accelerated sampling framework with autoencoder-based method](https://github.com/weiHelloWorld/accelerated_sampling_with_autoencoder)

# QM/MM
- NNP-MM(https://github.com/RowleyGroup/NNP-MM)
<br>NNP/MM embeds a Neural Network Potential into a conventional molecular mechanical (MM) model. We have implemented this using the Custom QM/MM features of NAMD 2.13, which interface NAMD with the TorchANI NNP python library developed by the Roitberg and Isayev groups.
- DeeP-HP(https://github.com/TinkerTools/tinker-hp/tree/Deep-HP)
<br>support for neural networks potentials (ANI-2X, DeepMD)
