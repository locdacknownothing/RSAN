python3 train_drive.py --weight_name Model_Cl_1806 --epochs 100 --lr 0.001 --loss_name soft_dice_cldice --patience 150
python3 train_drive.py --weight_name Model_Cl_1806 --restore --epochs 150 --lr 0.0001 --loss_name soft_dice_cldice --patience 150
python3 eval_drive.py --weight_name Model_Cl_1806 
