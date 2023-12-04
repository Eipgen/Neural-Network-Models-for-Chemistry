# Neural-Network-Models-for-Chemistry
[![Check Markdown links](https://github.com/DiracMD/Neural-Network-Models-for-chemistry/actions/workflows/main.yml/badge.svg)](https://github.com/DiracMD/Neural-Network-Models-for-chemistry/actions/workflows/main.yml)

A collection of Neural Network Models for chemistry
- [Density Functional Theory Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#density-functional-theory-method)
- [Molecular Force Field Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#molecular-force-field-method)
- [Semi-Empirical Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Semi-Empirical-Quantum-Mechanical-Method)
- [Coarse-Grained Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Coarse-Grained-Method)
- [Enhanced Sampling Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Enhanced-Sampling-Method)
- [QM/MM Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#QMMM-Model)
- [Charge Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Charge-Model)
## Density Functional Theory Method

- [DeePKS, DeePHF](https://github.com/deepmodeling/deepks-kit)  
DeePKS-kit is a program to generate accurate energy functionals for quantum chemistry systems, for both perturbative scheme (DeePHF) and self-consistent scheme (DeePKS).

- [NeuralXC](https://github.com/semodi/neuralxc)  
Implementation of a machine learned density functional.

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
JAX-DFT implements one-dimensional density functional theory (DFT) in JAX. It uses powerful JAX primitives to enable JIT compilation, automatical differentiation, and high-performance computation on GPUs.
- [sns-mp2](https://github.com/DEShawResearch/sns-mp2)  
Improving the accuracy of Moller-Plesset perturbation theory with neural networks
- [DeepH-pack](https://github.com/mzjb/DeepH-pack)  
Deep neural networks for density functional theory Hamiltonian.
- [kdft](https://gitlab.com/jmargraf/kdf)  
The Kernel Density Functional (KDF) code allows generating ML based DFT functionals.
- [ML-DFT](https://github.com/MihailBogojeski/ml-dft)  
ML-DFT: Machine learning for density functional approximations This repository contains the implementation for the kernel ridge regression based density functional approximation method described in the paper "Quantum chemical accuracy from density functional approximations via machine learning".
- [D4FT](https://github.com/sail-sg/d4ft)  
this work proposed a deep learning approach to KS-DFT. First, in contrast to the conventional SCF loop, directly minimizing the total energy by reparameterizing the orthogonal constraint as a feed-forward computation. They prove that such an approach has the same expressivity as the SCF method yet reduces the computational complexity from O(N^4) to O(N^3)
- [SchOrb](https://github.com/atomistic-machine-learning/SchNOrb)  
Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions
- [CiderPress](https://github.com/mir-group/CiderPress)  
Tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations.
<!-- markdown-link-check-disable-next-line -->
- [ML-RPA](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00848)  
This work demonstrates how machine learning can extend the applicability of the RPA to larger system sizes, time scales, and chemical spaces.

## Molecular Force Field Method

- [DeePMD](https://github.com/deepmodeling/deepmd-kit)
<br>A package designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics.
- [Torch-ANI](https://github.com/aiqm/torchani)
<br>TorchANI is a pytorch implementation of ANI model.
- [mdgrad](https://github.com/torchmd/mdgrad/tree/pair)  
Pytorch differentiable molecular dynamics
- [Schrodinger-ANI](http://public-sani.onschrodinger.com/)  
A neural network potential energy function for use in drug discovery, with chemical element support extended from 41% to 94% of druglike molecules based on ChEMBL.
- [NerualForceFild](https://github.com/learningmatter-mit/NeuralForceField)
<br>The Neural Force Field (NFF) code is an API based on SchNet, DimeNet, PaiNN and DANN . It provides an interface to train and evaluate neural networks for force fields. It can also be used as a property predictor that uses both 3D geometries and 2D graph information.
- [NNPOps](https://github.com/openmm/NNPOps)
<br>The goal of this project is to promote the use of neural network potentials (NNPs) by providing highly optimized, open source implementations of bottleneck operations that appear in popular potentials.
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
- [Nequip](https://github.com/mir-group/nequip)
<br>NequIP is an open-source code for building E(3)-equivariant interatomic potentials.
- [E3NN](https://github.com/e3nn/e3nn)  
Euclidean neural networks,The aim of this library is to help the development of E(3) equivariant neural networks. It contains fundamental mathematical operations such as tensor products and spherical harmonics.
- [SchNet](https://github.com/atomistic-machine-learning/SchNet)
<br>SchNet is a deep learning architecture that allows for spatially and chemically resolved insights into quantum-mechanical observables of atomistic systems.
- [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
<br>SchNetPack aims to provide accessible atomistic neural networks that can be trained and applied out-of-the-box, while still being extensible to custom atomistic architectures.
contains `schnet`,`painn`,`filedschnet`,`so3net`
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
- [AIMNET](https://github.com/aiqm/aimnet)  
This repository contains reference AIMNet implementation along with some examples and menchmarks.
- [charge_transfer_nnp](https://github.com/pfnet-research/charge_transfer_nnp)  
About Graph neural network potential with charge transfer with nequip model.
- [AMP](https://amp.readthedocs.io/en/latest/)  
Amp: A modular approach to machine learning in atomistic simulations(<https://github.com/ulissigroup/amptorch>)
- [SCFNN](https://github.com/andy90/SCFNN)  
A self consistent field neural network (SCFNN) model.
- [jax-md](https://github.com/google/jax-md)  
 JAX MD is a functional and data driven library. Data is stored in arrays or tuples of arrays and functions transform data from one state to another.
- [EANN](https://github.com/zhangylch/EANN)  
Embedded Atomic Neural Network (EANN) is a physically-inspired neural network framework. The EANN package is implemented using the PyTorch framework used to train interatomic potentials, dipole moments, transition dipole moments and polarizabilities of various systems.
- [REANN](https://github.com/zhangylch/REANN)  
 Recursively embedded atom neural network (REANN) is a PyTorch-based end-to-end multi-functional Deep Neural Network Package for Molecular, Reactive and Periodic Systems.
- [FIREANN](https://github.com/zhangylch/FIREANN)  
 Field-induced Recursively embedded atom neural network (FIREANN) is a PyTorch-based end-to-end multi-functional Deep Neural Network Package for Molecular, Reactive and Periodic Systems under the presence of the external field with rigorous rotational equivariance. 
- [espaloma](https://github.com/choderalab/espaloma)  
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm.
- [MDsim](https://github.com/kyonofx/MDsim)  
Training and simulating MD with ML force fields
- [ForceNet](https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/forcenet.py)   
We demonstrate that force-centric GNN models without any explicit physical constraints are able to predict atomic forces more accurately than state-of-the-art energy centric GNN models, while being faster both in training and inference.
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
- [flare](https://github.com/mir-group/flare)  
FLARE is an open-source Python package for creating fast and accurate interatomic potentials.)  
- [alignn](https://github.com/usnistgov/alignn)  
The Atomistic Line Graph Neural Network (https://www.nature.com/articles/s41524-021-00650-1) introduces a new graph convolution layer that explicitly models both two and three body interactions in atomistic systems.
- [So3krates](https://github.com/thorben-frank/mlff)  
Repository for training, testing and developing machine learned force fields using the So3krates model. 
- [spice-model-five-net](https://github.com/openmm/spice-models/tree/main/five-et)  
Contains the five equivariant transformer models about the spice datasets(https://github.com/openmm/spice-dataset/releases/tag/1.1). 
- [sake](https://github.com/choderalab/sake)  
Spatial Attention Kinetic Networks with E(n)-Equivariance
- [eqgat](https://github.com/Bayer-Group/eqgat)  
Pytorch implementation for the manuscript Representation Learning on Biomolecular Structures using Equivariant Graph Attention
- [phast](https://github.com/vict0rsch/phast)  
PyTorch implementation for PhAST: Physics-Aware, Scalable and Task-specific GNNs for Accelerated Catalyst Design
- [GNN-LF](https://github.com/GraphPKU/GNN-LF)  
Graph Neural Network With Local Frame for Molecular Potential Energy Surface  
- [Cormorant](https://arxiv.org/abs/1906.04015)  
We propose Cormorant, a rotationally covariant neural network architecture for learning the behavior and properties of complex many-body physical systems.
- [LieConv](https://github.com/mfinzi/LieConv)  
Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data
- [torchmd-net/ET](https://github.com/torchmd/torchmd-net)  
Neural network potentials based on graph neural networks and equivariant transformers
- [GemNet](https://github.com/TUM-DAML/gemnet_tf)  
GemNet: Universal Directional Graph Neural Networks for Molecules
- [equiformer](https://github.com/atomicarchitects/equiformer)  
Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs
- [VisNet-LSRM](https://arxiv.org/abs/2304.13542)  
Inspired by fragmentation-based methods, we propose the Long-Short-Range Message-Passing (LSR-MP) framework as a generalization of the existing equivariant graph neural networks (EGNNs) with the intent to incorporate long-range interactions efficiently and effectively. 
- [AP-net](https://github.com/zachglick/AP-Net)  
AP-Net: An atomic-pairwise neural network for smooth and transferable interaction potentials
- [mace](https://github.com/ACEsuit/mace)  
MACE provides fast and accurate machine learning interatomic potentials with higher order equivariant message passing.
- [Unimol+](https://arxiv.org/pdf/2303.16982.pdf)
Uni-Mol+ first generates a raw 3D molecule conformation from inexpensive methods such as RDKit. Then, the raw conformation is iteratively updated to its target DFT equilibrium conformation using neural networks, and the learned conformation will be used to predict the QC properties.
- [ColfNet](https://proceedings.mlr.press/v162/du22e/du22e.pdf)  
Inspired by differential geometry and physics, we introduce equivariant local complete frames to graph neural networks, such that tensor information at given orders can be projected onto the frames.
- [AIRS](https://github.com/divelab/AIRS)  
AIRS is a collection of open-source software tools, datasets, and benchmarks associated with our paper entitled “Artificial Intelligence for Science in Quantum, Atomistic, and Continuum Systems”.
- [nnp-pre-training](https://github.com/jla-gardner/nnp-pre-training)  
  Synthetic pre-training for neural-network interatomic potentials
- [AlF_dimer](https://github.com/onewhich/AlF_dimer)  
  a global potential for AlF-AlF dimer
- [q-AQUA,q-AQUA-pol](https://github.com/jmbowma/q-AQUA)  
  CCSD(T) potential for water, interfaced with TTM3-F
- [LeftNet](https://github.com/yuanqidu/LeftNet)  
  A New Perspective on Building Efficient and Expressive 3D Equivariant Graph Neural Networks
- [mlp-train](https://github.com/duartegroup/mlp-train)   
  General machine learning potentials (MLP) training for molecular systems in gas phase and solution
- [ARROW-NN](https://github.com/freecurve/interx_arrow-nn_suite)  
  The simulation conda package contains the InterX ARBALEST molecular dynamics simulation software along with all the necessary database files to run ARROW-NN molecular simulations
- [SO3krates with transformer](https://arxiv.org/abs/2309.15126)  
  we propose a transformer architecture called SO3krates that combines sparse equivariant representations
- [AMOEBA+NN](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/655532b7dbd7c8b54b56a3a0/original/incorporating-neural-networks-into-the-amoeba-polarizable-force-field.pdf)  
  It present an integrated non-reactive hybrid model, AMOEBA+NN, which employs the AMOEBA potential for the short- and long-range non-bonded interactions and an NNP to capture the remaining local (covalent) contributions
- [LEIGNN](https://github.com/guaguabujianle/LEIGNN)
  A lightweight equivariant interaction graph neural network (LEIGNN) that can enable accurate and efficient interatomic potential and force predictions in crystals. Rather than relying on higher-order representations, LEIGNN employs a scalar-vector dual representation to encode equivariant feature.
- [Arrow NN](https://github.com/freecurve/interx_arrow-nn_suite)  
  A hybrid wide-coverage intermolecular interaction model consisting of an analytically polarizable force field combined with a short-range neural network correction for the total intermolecular interaction energy. 

## Semi-Empirical Quantum Mechanical Method

- [OrbNet](https://arxiv.org/abs/2107.00299)
<br>OrbNet Denali: A machine learning potential for biological and organic chemistry with semi-empirical cost and DFT accuracy.

- [AIQM1](https://doi.org/10.1038/s41467-021-27340-2)
<br>Artificial intelligence-enhanced quantum chemical method with broad applicability.
<!-- markdown-link-check-disable-next-line -->
- [BpopNN](https://doi.org/10.1021/acs.jctc.0c00217)
<br>Incorporating Electronic Information into Machine Learning Potential Energy Surfaces via Approaching the Ground-State Electronic Energy as a Function of Atom-Based Electronic Populations.

- [Delfta](https://github.com/josejimenezluna/delfta)
<br>The DelFTa application is an easy-to-use, open-source toolbox for predicting quantum-mechanical properties of drug-like molecules. Using either ∆-learning (with a GFN2-xTB baseline) or direct-learning (without a baseline), the application accurately approximates DFT reference values (ωB97X-D/def2-SVP).
- [PYSEQM](https://github.com/lanl/PYSEQM)  
PYSEQM is a Semi-Empirical Quantum Mechanics package implemented in PyTorch.
- [DFTBML](https://github.com/djyaron/DFTBML)  
DFTBML provides a systematic way to parameterize the Density Functional-based Tight Binding (DFTB) semiempirical quantum chemical method for different chemical systems by learning the underlying Hamiltonian parameters rather than fitting the potential energy surface directly.

## Coarse-Grained Method

- [cgnet](https://github.com/coarse-graining/cgnet)
<br>Coarse graining for molecular dynamics
- [SchNet-CG](https://arxiv.org/ftp/arxiv/papers/2209/2209.12948.pdf)  
We explore the application of SchNet models to obtain a CG potential for liquid benzene, investigating the effect of model architecture and hyperparameters on the thermodynamic, dynamical, and structural properties of the simulated CG systems, reporting and discussing challenges encountered and future directions envisioned.

## Enhanced Sampling Method
<!-- markdown-link-check-disable-next-line -->
- [VES-NN](https://doi.org/10.1073/pnas.1907975116)  
Neural networks-based variationallyenhanced sampling

- [Accelerated_sampling_with_autoencoder](https://github.com/weiHelloWorld/accelerated_sampling_with_autoencoder)  
Accelerated sampling framework with autoencoder-based method
<!-- markdown-link-check-disable-next-line -->
- [Enhanced Sampling with Machine Learning: A Review](https://arxiv.org/pdf/2306.09111v2.pdf)  
we highlight successful strategies like dimensionality reduction, reinforcement learning, and fl ow-based methods. Finally, we discuss open problems at the exciting ML-enhanced MD interface
- [mlcolvar](https://github.com/luigibonati/mlcolvar)  
mlcolvar is a Python library aimed to help design data-driven collective-variables (CVs) for enhanced sampling simulations. 
## QM/MM Model

- [NNP-MM](https://github.com/RowleyGroup/NNP-MM)
<br>NNP/MM embeds a Neural Network Potential into a conventional molecular mechanical (MM) model. We have implemented this using the Custom QM/MM features of NAMD 2.13, which interface NAMD with the TorchANI NNP python library developed by the Roitberg and Isayev groups.
- [DeeP-HP](https://github.com/TinkerTools/tinker-hp/blob/master/GPU/Deep-HP.md)
<br> Scalable hybrid deep neural networks/polarizable potentials biomolecular simulations including long-range effects

## Charge Model

- [gimlet](https://github.com/choderalab/gimlet)  
Graph Inference on Molecular Topology. A package for modelling, learning, and inference on molecular topological space written in Python and TensorFlow.

## Post-HF Method
<!-- markdown-link-check-disable-next-line -->
- [LNO-CCSD(T)](https://pubs.acs.org/doi/10.1021/acs.jpca.0c11270)
<!-- markdown-link-check-disable-next-line -->
- [DLPNO-CCSD(T)](https://doi.org/10.1021/acs.jpca.0c11270)
<!-- markdown-link-check-disable-next-line -->
- [PNO-CCSD(T)](https://doi.org/10.1021/acs.jctc.7b00180)
