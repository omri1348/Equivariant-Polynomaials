conda activate poly_env

python main.py lr=2e-3 lr_schedule_patience=10 depth=8 out_dim=1 hidden_dim=95 input_dim=3 dataset=ZINC seed=41 &
sleep 30s
python main.py lr=2e-3 lr_schedule_patience=10 depth=8 out_dim=1 hidden_dim=95 input_dim=3 dataset=ZINC seed=95 &
sleep 30s
python main.py lr=2e-3 lr_schedule_patience=10 depth=8 out_dim=1 hidden_dim=95 input_dim=3 dataset=ZINC seed=35 &
sleep 30s
python main.py lr=2e-3 lr_schedule_patience=10 depth=8 out_dim=1 hidden_dim=95 input_dim=3 dataset=ZINC seed=12