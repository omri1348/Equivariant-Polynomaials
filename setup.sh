conda activate poly_env

python equivariant_polynomials.py
mkdir data/ZINC/subset_poly
mkdir data/ZINC/subset_poly/ppgn
mkdir data/Alchemy/poly
mkdir data/Alchemy/poly/ppgn
mkdir data/SR/poly
mkdir data/SR/poly/ppgn
python dataset.py

