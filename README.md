# Neural-Network-Models-for-Chemistry
[![Check Markdown links](https://github.com/DiracMD/Neural-Network-Models-for-chemistry/actions/workflows/main.yml/badge.svg)](https://github.com/DiracMD/Neural-Network-Models-for-chemistry/actions/workflows/main.yml)

A collection of Neural Network Models for chemistry
- [Quantum Chemistry Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#quantum-chemistry-method)
- [Force Field Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#force-field-method)
- [Semi-Empirical Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Semi-Empirical-Quantum-Mechanical-Method)
- [Coarse-Grained Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Coarse-Grained-Method)
- [Enhanced Sampling Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Enhanced-Sampling-Method)
- [QM/MM Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#QMMM-Model)
- [Charge Method](https://github.com/Eipgen/Neural-Network-Models-for-Chemistry/blob/main/README.md#Charge-Model)
## Quantum Chemistry Method

- [DeePKS, DeePHF](https://github.com/deepmodeling/deepks-kit)  
DeePKS-kit is a program to generate accurate energy functionals for quantum chemistry systems, for both perturbative scheme (DeePHF) and self-consistent scheme (DeePKS).

- [NeuralXC](https://github.com/semodi/neuralxc)  
Implementation of a machine-learned density functional.
<!-- markdown-link-check-disable-next-line -->
- [MOB-ML](https://aip.scitation.org/doi/10.1063/5.0032362)   
Machine Learning for Molecular Orbital Theory, they offer analytic gradient.

- [DM21](https://github.com/deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)  
Pushing the Frontiers of Density Functionals by Solving the Fractional Electron Problem.
- [NN-GGA, NN-NRA, NN-meta-GGA, NN-LSDA](https://github.com/ml-electron-project/NNfunctional)  
Completing density functional theory by machine-learning hidden messages from molecules.
- [FemiNet](https://github.com/deepmind/ferminet)  
FermiNet is a neural network for learning highly accurate ground state wavefunctions of atoms and molecules using a variational Monte Carlo approach.
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
-[DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3)  
General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian
- [kdft](https://gitlab.com/jmargraf/kdf)  
The Kernel Density Functional (KDF) code allows generating ML-based DFT functionals.
- [ML-DFT](https://github.com/MihailBogojeski/ml-dft)  
ML-DFT: Machine learning for density functional approximations This repository contains the implementation for the kernel ridge regression based density functional approximation method described in the paper "Quantum chemical accuracy from density functional approximations via machine learning".
- [D4FT](https://github.com/sail-sg/d4ft)  
this work proposed a deep-learning approach to KS-DFT. First, in contrast to the conventional SCF loop, directly minimizing the total energy by reparameterizing the orthogonal constraint as a feed-forward computation. They prove that such an approach has the same expressivity as the SCF method yet reduces the computational complexity from O(N^4) to O(N^3)
- [SchOrb](https://github.com/atomistic-machine-learning/SchNOrb)  
Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions
<!-- markdown-link-check-disable-next-line -->
- [CiderPress](https://github.com/mir-group/CiderPress)  
Tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations.
<!-- markdown-link-check-disable-next-line -->
- [ML-RPA](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00848)  
This work demonstrates how machine learning can extend the applicability of the RPA to larger system sizes, time scales, and chemical spaces.
<!-- markdown-link-check-disable-next-line -->
- [ΔOF-MLFF](https://pubs.aip.org/aip/jcp/article/159/24/244106/2931521/Kohn-Sham-accuracy-from-orbital-free-density)  
a Δ-machine learning model for obtaining Kohn–Sham accuracy from orbital-free density functional theory (DFT) calculations
<!-- markdown-link-check-disable-next-line -->
- [PairNet](https://doi.org/10.26434/chemrxiv-2023-n1skn)   
A molecular orbital based machine learning model for predicting accurate CCSD(T) correlation energies. The model, named as PairNet, shows excellent transferability on several public data sets using features inspired by pair natural orbitals(PNOs).

- [SPAHM(a,b)](https://github.com/lcmd-epfl/SPAHM-RHO)  
SPAHM(a,b): encoding the density information from guess Hamiltonian in quantum machine learning representations
- [GradDFT](https://github.com/XanaduAI/GradDFT)  
GradDFT is a JAX-based library enabling the differentiable design and experimentation of exchange-correlation functionals using machine learning techniques.
- [lapnet](https://github.com/bytedance/LapNet)  
A JAX implementation of the algorithm and calculations described in Forward Laplacian: A New Computational Framework for Neural Network-based Variational Monte Carlo.
- [M-OFDFT](https://zenodo.org/records/10616893)  
M-OFDFT is a deep-learning implementation of orbital-free density functional theory that achieves DFT-level accuracy on molecular systems but with lower cost complexity, and can extrapolate to much larger molecules than those seen during training
- [ACE-Kohn-Sham DM](https://arxiv.org/pdf/2503.08400)
  
<!-- markdown-link-check-disable-next-line -->
- [ANN for Schrodinger](https://doi.org/10.26434/chemrxiv-2024-2qw5x)  
  Artificial neural networks (NN) are universal function approximators and have shown great ability in computing the ground state energy of the electronic Schrödinger equation, yet NN has not established itself as a practical and accurate approach to solving the vibrational Schrödinger equation for realistic polyatomic molecules to obtain vibrational energies and wave functions for the excited states
- [equivariant_electron_density](https://github.com/JoshRackers/equivariant_electron_density)  
  Generate and predict molecular electron densities with Euclidean Neural Networks
- [DeePDFT](https://github.com/peterbjorgensen/DeepDFT)  
This is the official Implementation of the DeepDFT model for charge density prediction.
- [DFA_recommeder](https://github.com/hjkgrp/dfa_recommender)  
  System-specific density functional recommender
- [EG-XC](https://arxiv.org/pdf/2410.07972v1)  
The accuracy of density functional theory hinges on the approximation of nonlocal contributions to the exchange-correlation (XC) functional. To date, machine-learned and human-designed approximations suffer from insufficient accuracy, limited scalability, or dependence on costly reference data. To address these issues, we introduce Equivariant Graph Exchange Correlation (EG-XC), a novel non-local XC functional based on equivariant graph neural network
- [scdp](https://github.com/kyonofx/scdp)  
Machine learning methods are promising in significantly accelerating charge density prediction, yet existing approaches either lack accuracy or scalability. They  propose a recipe that can achieve both. In particular, they identify three key ingredients: (1) representing the charge density with atomic and virtual orbitals (spherical fields centered at atom/virtual coordinates); (2) using expressive and learnable orbital basis sets (basis function for the spherical fields); and (3) using high-capacity equivariant neural network architecture
- [physics-informed-DFT](https://github.com/TheorChemGroup/physics-informed-DFT)  
We have developed an approach for physics-informed training of flexible empirical density functionals. In this approach, the “physics knowledge” is transferred from PBE, or any other exact-constraints-based functional, using local exchange−correlation energy density regularization, i.e., by adding its local energies into the training set
- [SchrodingerNet](https://github.com/zhangylch/SchrodingerNet)   
 SchrödingerNet offers a novel approach to solving the full electronic-nuclear Schrödinger equation (SE) by defining a custom loss function designed to equalize local energies throughout the system.
- [qmlearn](https://gitlab.com/pavanello-research-group/qmlearn)  
  Quantum Machine Learning by learning one-body reduced density matrices in the AO basis.
- [Multi-task-electronic](https://github.com/htang113/Multi-task-electronic)  
  This package provides a python realization of the multi-task EGNN (equivariant graph neural network) for molecular electronic structure described in the paper "Multi-task learning for molecular electronic structure approaching coupled-cluster accuracy".
- [aPBE0](https://github.com/dkhan42/aPBE0)  
   We propose adaptive hybrid functionals, generating optimal exact exchange admixture ratios on the fly using data- efficient quantum  machine  learning  models  with  negligible  overhead. The  adaptive  Perdew-Burke-Ernzerhof  hybrid  density functional (aPBE0) improves energetics, electron densities, and HOMO- LUMO gaps in QM9, QM7b, and GMTKN55 benchmark datasets.

   
### Quantum Monte Carlo
- [DeePQMC](https://github.com/deepqmc/deepqmc)  
DeepQMC implements variational quantum Monte Carlo for electrons in molecules, using deep neural networks written in PyTorch as trial wave functions.


### Green Function
- [DeepGreen](https://arxiv.org/abs/2312.14680)  
The many-body Green's function provides access to electronic properties beyond density functional theory level in ab inito calculations. It present proof-of-concept benchmark results for both molecules and simple periodic systems, showing that our method is able to provide accurate estimate of physical observables such as energy and density of states based on the predicted Green's function.
## Quantum Monte Carlo




## Force Field Method

### Kernel Method
- [wigner_kernel](https://github.com/lab-cosmo/wigner_kernels.git)  
  They propose a novel density-based method which involves computing “Wigner kernels”. 
###  local descriptor-based model
- [Torch-ANI](https://github.com/aiqm/torchani)
<br>TorchANI is a pytorch implementation of ANI model.
- [PESPIP](https://github.com/PaulLHouston/PESPIP)  
Mathematica programs for choosing the best basis of permutational invariant polynomials for fitting a potential energy surface
- [RuNNer](https://www.uni-goettingen.de/de/software/616512.html)
<br>A program package for constructing high-dimensional neural network potentials,4G-HDNNPs,3G-HDNNPs.
- [sGDML](http://www.sgdml.org/)
<br> Symmetric Gradient Domain Machine Learning
###  invariant model
- [DeePMD](https://github.com/deepmodeling/deepmd-kit)
<br>A package designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics.
###  equivariant model
###  universal model

### others
- [mdgrad](https://github.com/torchmd/mdgrad/tree/pair)  
Pytorch differentiable molecular dynamics
<!-- markdown-link-check-disable-next-line -->
- [Schrodinger-ANI](http://public-sani.onschrodinger.com/)  
A neural network potential energy function for use in drug discovery, with chemical element support extended from 41% to 94% of druglike molecules based on ChEMBL.
- [NerualForceFild](https://github.com/learningmatter-mit/NeuralForceField)
<br>The Neural Force Field (NFF) code is an API based on SchNet, DimeNet, PaiNN and DANN. It provides an interface to train and evaluate neural networks for force fields. It can also be used as a property predictor that uses both 3D geometries and 2D graph information.
- [NNPOps](https://github.com/openmm/NNPOps)
<br>The goal of this project is to promote the use of neural network potentials (NNPs) by providing highly optimized, open-source implementations of bottleneck operations that appear in popular potentials.
- [aenet](https://github.com/atomisticnet/aenet)
<br>The Atomic Energy NETwork (ænet) package is a collection of tools for the construction and application of atomic interaction potentials based on artificial neural networks.
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
- [AIMNet2](https://github.com/isayevlab/AIMNet2)  
A general-purpose neural netrork potential for organic and element-organic molecules.
- [aevmod](https://github.com/sandialabs/aevmod)  
This package provides functionality for computing an atomic environment vector (AEV), as well as its Jacobian and Hessian.
- [charge3net](https://github.com/AIforGreatGood/charge3net)   
Official implementation of ChargeE3Net, introduced in Higher-Order Equivariant Neural Networks for Charge Density Prediction in Materials.
- [jax-nb](https://github.com/reaxnet/jax-nb)  
  This is a JAX implementation of Polarizable Charge Equilibrium (PQEq) and DFT-D3 dispersion correction.
### Graph Domain
- [Nequip](https://github.com/mir-group/nequip)
<br>NequIP is an open-source code for building E(3)-equivariant interatomic potentials.
- [E3NN](https://github.com/e3nn/e3nn)  
Euclidean neural networks,The aim of this library is to help the development of E(3) equivariant neural networks. It contains fundamental mathematical operations such as tensor products and spherical harmonics.
- [SchNet](https://github.com/atomistic-machine-learning/SchNet)
<br>SchNet is a deep learning architecture that allows for spatially and chemically resolved insights into quantum-mechanical observables of atomistic systems.
- [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack)
<br>SchNetPack aims to provide accessible atomistic neural networks that can be trained and applied out-of-the-box, while still being extensible to custom atomistic architectures.
contains `schnet`,`painn`,`filedschnet`,`so3net`
- [XequiNet](https://github.com/X1X1010/XequiNet)
  XequiNet is an equivariant graph neural network for predicting the properties of chemical molecules or periodical systems.
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
- [MDsim](https://github.com/kyonofx/MDsim)  
Training and simulating MD with ML force fields
- [ForceNet](https://github.com/Open-Catalyst-Project/ocp)   
We demonstrate that force-centric GNN models without any explicit physical constraints are able to predict atomic forces more accurately than state-of-the-art energy centric GNN models, while being faster both in training and inference.
- [DIG](https://github.com/divelab/DIG)  
A library for graph deep learning research.
- [scn](https://github.com/Open-Catalyst-Project/ocp)  
Spherical Channels for Modeling Atomic Interactions
- [spinconv](https://github.com/Open-Catalyst-Project/ocp)  
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
- [torchmd-net/TensorNet+0.1S](https://github.com/torchmd/torchmd-net)  
  On the Inclusion of Charge and Spin States in Cartesian Tensor Neural Network Potentials
- [GemNet](https://github.com/TUM-DAML/gemnet_tf)  
GemNet: Universal Directional Graph Neural Networks for Molecules
- [equiformer](https://github.com/atomicarchitects/equiformer)  
Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs
- [VisNet-LSRM](https://arxiv.org/abs/2304.13542)  
Inspired by fragmentation-based methods, we propose the Long-Short-Range Message-Passing (LSR-MP) framework as a generalization of the existing equivariant graph neural networks (EGNNs) with the intent to incorporate long-range interactions efficiently and effectively. 
- [AP-net](https://github.com/zachglick/AP-Net)  
AP-Net: An atomic-pairwise neural network for smooth and transferable interaction potentials
- [MACE](https://github.com/ACEsuit/mace)  
MACE provides fast and accurate machine learning interatomic potentials with higher order equivariant message passing.
- [MACE-OFF23](https://github.com/ACEsuit/mace-off)  
 This repository contains the MACE-OFF23 pre-traained transferable organic force fields. 
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
- [PAMNet](https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN)  
PAMNet(Physics-aware Multiplex Graph Neural Network) is an improved version of MXMNet and outperforms state-of-the-art baselines regarding both accuracy and efficiency in diverse tasks including small molecule property prediction, RNA 3D structure prediction, and protein-ligand binding affinity prediction.
- [Multi-fidelity GNNs](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics)  
Multi-fidelity GNNs for drug discovery and quantum mechanics
- [GPIP](https://github.com/cuitaoyong/GPIP)  
GPIP: Geometry-enhanced Pre-training on Interatomic Potentials.they propose a geometric structure learning framework that leverages the unlabeled configurations to improve the performance of MLIPs. Their framework consists of two stages: firstly, using CMD simulations to generate unlabeled configurations of the target molecular system; and secondly, applying geometry-enhanced self-supervised learning techniques, including masking, denoising, and contrastive learning, to capture structural information 
- [ictp](https://github.com/nec-research/ictp)  
Official repository for the paper "Higher Rank Irreducible Cartesian Tensors for Equivariant Message Passing". It is built upon the ALEBREW repository and implements irreducible Cartesian tensors and their products.

- [CHGNet](https://github.com/CederGroupHub/chgnet)  
A pretrained universal neural network potential for charge-informed atomistic modeling (see publication) 
- [GPTFF](https://arxiv.org/abs/2402.19327)  
GPTFF: A high-accuracy out-of-the-box universal AI force field for arbitrary inorganic materials
- [rascaline](https://github.com/Luthaf/rascaline)  
Rascaline is a library for the efficient computing of representations for atomistic machine learning also called "descriptors" or "fingerprints". These representations can be used for atomistic machine learning (ml) models including ml potentials, visualization or similarity analysis.
<!-- markdown-link-check-disable-next-line -->
- [PairNet-OPs/PairFE-Net](https://doi.org/10.1039/D4SC01109K)  
In PairFE-Net, an atomic structure is encoded using pairwise nuclear repulsion forces

- [bamboo](https://github.com/bytedance/bamboo)  
ByteDance AI Molecular Simulation BOOster (BAMBOO)
- [cace](https://github.com/BingqingCheng/cace)  
The Cartesian Atomic Cluster Expansion (CACE) is a new approach for developing machine learning interatomic potentials. This method utilizes Cartesian coordinates to provide a complete description of atomic environments, maintaining interaction body orders. It integrates low-dimensional embeddings of chemical elements with inter-atomic message passing.
### Transformer Domain
- [SpookyNet](https://github.com/OUnke/SpookyNet)
<br>Spookynet: Learning force fields with electronic degrees of freedom and nonlocal effects.
- [trip](https://github.com/dellacortelab/trip)  
Transformer Interatomic Potential (TrIP): a chemically sound potential based on the SE(3)-Transformer
- [e3x](https://github.com/google-research/e3x)  
E3x is a JAX library for constructing efficient E(3)-equivariant deep learning architectures built on top of Flax. The goal is to provide common neural network building blocks for E(3)-equivariant architectures to make the development of models operating on three-dimensional data (point clouds, polygon meshes, etc.) easier.
- [EScAIP](https://github.com/ASK-Berkeley/EScAIP)   
  EScAIP: Efficiently Scaled Attention Interatomic Potential.
- [eSEN](https://github.com/facebookresearch/fairchem/tree/be0ea9cdf08ad00ce1d65ba69680129965294320/src/fairchem/core/models/esen)  
  The resulting model, eSEN, provides state-of-the-art results on a range of physical property prediction tasks,
- [UNA](https://huggingface.co/facebook/UMA)  
  UMA: A Family of Universal Models for Atoms, a modified model of eSEN.
### Empirical force field

- [grappa](https://github.com/graeter-group/grappa)  
  A machine-learned molecular mechanics force field using a deep graph attentional network
- [espaloma](https://github.com/choderalab/espaloma)  
Extensible Surrogate Potential of Ab initio Learned and Optimized by Message-passing Algorithm.
- [FeNNol](https://github.com/thomasple/FeNNol)  
FeNNol is a library for building, training and running neural network potentials for molecular simulations. It is based on the JAX library and is designed to be fast and flexible.
- [ByteFF](https://arxiv.org/abs/2408.12817)  
In this study, we address this issue usinga modern data-driven approach, developing ByteFF, an Amber-compatible force fi eld for drug-like molecules. To create ByteFF, we generated an expansive and highly diverse molecular dataset at the B3LYP-D3(BJ)/DZVP level of theory. This dataset includes 2.4 million optimized molecular fragment geometries with analytical Hessian matrices, along with 3.2 million torsion profiles
- [GB-FFs](https://github.com/GongCHEN-1995/GB-FFs-Model)  
  Graph-Based Force Fields Model to parameterize Force Fields by Graph Attention Networks
## Semi-Empirical Quantum Mechanical Method
#### with SQM feature
- [OrbNet; OrbNet Denali](https://arxiv.org/abs/2107.00299)
<br>OrbNet Denali: A machine learning potential for biological and organic chemistry with semi-empirical cost and DFT accuracy.
<!-- markdown-link-check-disable-next-line -->
- [ OrbNet-Equi](https://doi.org/10.1073/pnas.2205221119)  
INFORMING GEOMETRIC DEEP LEARNING WITH ELECTRONIC INTERACTIONS TO ACCELERATE QUANTUM CHEMISTRY
<!-- markdown-link-check-disable-next-line -->
- [OrbNet-Spin](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_214.pdf)   
OrbNet-Spin incorporates a spin-polarized treatment into the underlying semiempirical quantum mechanics orbital featurization and adjusts the model architecture accordingly while maintaining the geometrical constraints.
- [EHM-ML](https://github.com/aiqm/EHM-ML)     
Machine Learned Hückel Theory: Interfacing Physics and Deep Neural Networks. The Hückel Hamiltonian is an incredibly simple tight-binding model known for its ability to capture qualitative physics phenomena arising from electron interactions in molecules and materials.
- [DFTBML](https://github.com/djyaron/DFTBML)  
DFTBML provides a systematic way to parameterize the Density Functional-based Tight Binding (DFTB) semiempirical quantum chemical method for different chemical systems by learning the underlying Hamiltonian parameters rather than fitting the potential energy surface directly.
#### without SQM fearure
- [AIQM1, AIQM2](https://doi.org/10.1038/s41467-021-27340-2) 
<br>Artificial intelligence-enhanced quantum chemical method with broad applicability.
<!-- markdown-link-check-disable-next-line -->
- [BpopNN](https://doi.org/10.1021/acs.jctc.0c00217) 
<br>Incorporating Electronic Information into Machine Learning Potential Energy Surfaces via Approaching the Ground-State Electronic Energy as a Function of Atom-Based Electronic Populations.
- [Delfta](https://github.com/josejimenezluna/delfta) 
<br>The DelFTa application is an easy-to-use, open-source toolbox for predicting quantum-mechanical properties of drug-like molecules. Using either ∆-learning (with a GFN2-xTB baseline) or direct-learning (without a baseline), the application accurately approximates DFT reference values (ωB97X-D/def2-SVP).
- [PYSEQM](https://github.com/lanl/PYSEQM)  
PYSEQM is a Semi-Empirical Quantum Mechanics package implemented in PyTorch.
- [PM6-ML](https://github.com/Honza-R/mopac-ml)  
MOPAC-ML implements the PM6-ML method, a semiempirical quantum-mechanical computational method that augments PM6 with a machine learning (ML) correction. It acts as a wrapper calling a modified version of MOPAC, to which it provides the ML correction.
- [XpaiNN@xTB](https://github.com/X1X1010/XequiNet)  
A model can deal with optimization, and frequency prediction

## Coarse-Grained Method

- [cgnet](https://github.com/coarse-graining/cgnet)
<br>Coarse graining for molecular dynamics
- [SchNet-CG](https://arxiv.org/ftp/arxiv/papers/2209/2209.12948.pdf)  
We explore the application of SchNet models to obtain a CG potential for liquid benzene, investigating the effect of model architecture and hyperparameters on the thermodynamic, dynamical, and structural properties of the simulated CG systems, reporting and discussing challenges encountered and future directions envisioned.
- [CG-SchNET](https://arxiv.org/pdf/2310.18278.pdf)  
By combining recent deep learning methods with a large and diverse training set of all-atom protein simulations, we here develop a bottom-up CG force field with chemical transferability, which can be used for extrapolative
molecular dynamics on new sequences not used during model parametrization.
- [torchmd-protein-thermodynamics](https://github.com/torchmd/torchmd-protein-thermodynamics)  
This repository contains code, data and tutarial for reproducing the paper "Machine Learning Coarse-Grained Potentials of Protein Thermodynamics". https://arxiv.org/abs/2212.07492
- [torchmd-exp](https://github.com/compsciencelab/torchmd-exp)     
This repository contains a method for training a neural network potential for coarse-grained proteins using unsupervised learning
- [AICG](https://github.com/jasonzzj97/AICG)  
Learning coarse-grained force fields for fibrogenesis modeling(https://doi.org/10.1016/j.cpc.2023.108964)



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
<!-- markdown-link-check-disable-next-line -->
- [PairF-Net](https://onlinelibrary.wiley.com/doi/10.1002/jcc.27313)  
 Here, we further develop the PairF-Net model to intrinsically incorporate energy conservation and couple the model to a molecular mechanical (MM) environment within the OpenMM package
- [embedding](https://github.com/emedio/embedding)  
 This work presents a variant of an electrostatic embedding scheme that allows the embedding of arbitrary machine learned potentials trained on molecular systems in vacuo.
- [field_schnet](https://github.com/atomistic-machine-learning/field_schnet)  
FieldSchNet provides a deep neural network for modeling the interaction of molecules and external environments as described.
- [FieldMACE](https://github.com/rhyan10/FieldMACE/tree/master)
  an extension of the message-passing atomic cluster expansion (MACE) architecture that integrates the multipole expansion to model long-range interactions more effi ciently. By incorporating the multipole expansion, FieldMACE eff ectively captures environmental and long-range eff ects in both ground and excited states.
- [MLMM](https://github.com/MALBECC/MLMM-embeddings-assessment-paper)  
  This repository contains data and software regarding the paper submited to JCIM, entitled "Assessment of embedding schemes in a hybrid machine learning/classical potentials (ML/MM) approach".
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
