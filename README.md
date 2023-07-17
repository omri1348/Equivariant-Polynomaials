# Equivariant-Polynomaials
Official implementation of the ICML 23 paper "Equivariant Polynomials for Graph Neural Networks" (Oral).

## Abstract
Graph Neural Networks (GNN) are inherently limited in their expressive power. Recent seminal works (Xu et al., 2019; Morris et al., 2019b) introduced the Weisfeiler-Lehman (WL) hierarchy as a measure of expressive power. Although this hierarchy has propelled significant advances in GNN analysis and architecture developments, it suffers from several significant limitations. These include a complex definition that lacks direct guidance for model improvement and a WL hierarchy that is too coarse to study current GNNs. This paper introduces an alternative expressive power hierarchy based on the ability of GNNs to calculate equivariant polynomials of a certain degree. As a first step, we provide a full characterization of all equivariant graph polynomials by introducing a concrete basis, significantly generalizing previous results. Each basis element corresponds to a specific multi-graph, and its computation over some graph data input corresponds to a tensor contraction problem. Second, we propose algorithmic tools for evaluating the expressiveness of GNNs using tensor contraction sequences, and calculate the expressive power of popular GNNs. Finally, we enhance the expressivity of common GNN architectures by adding polynomial features or additional operations / aggregations inspired by our theory. These enhanced GNNs demonstrate state-of-the-art results in experiments across multiple graph learning benchmarks.

For more details, see: [https://arxiv.org/abs/2302.11556](https://arxiv.org/abs/2302.11556).

## Installation Requirmenets
The code is compatible with python 3.9 and pytorch 1.12 (CUDA 10.2). Conda environment file is provided at ``env_spec_file.txt``.
To generate the environment run the following code:
```
onda create --name <env> --file env-spec-file.txt
```

## Usage
The repository includes implementations for the following experiments: ZINC (small), Alchemy, and the Strongly Regular (SR) expressiveness experiment. The provided example scripts replicate the results achieved by PPGN++. To reproduce these results, begin by activating the repository's environment.  
```
conda activate poly_env
```
Afterward, execute the setup file to download the data and precompute polynomials.
```
bash setup.sh
```
Lastly, execute the corresponding .sh file to run the experiment of your choice, for example, to run the ZINC experiment:
```
bash zinc.sh
```

## Citation 
```
@misc{puny2023equivariant,
      title={Equivariant Polynomials for Graph Neural Networks}, 
      author={Omri Puny and Derek Lim and Bobak T. Kiani and Haggai Maron and Yaron Lipman},
      year={2023},
      eprint={2302.11556},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```