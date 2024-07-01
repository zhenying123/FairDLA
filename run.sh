# ## best hyper-parameter for german dataset
# echo '============German============='
# CUDA_VISIBLE_DEVICES=0 python train.py --dropout 0.5 --hidden 16 --lr 1e-2 --epochs 1000 --model adagcn --dataset german --seed_num 5 --alpha 0.1 --beta 1.0

# ## best hyper-parameter for bail dataset
#echo '============Bail============='
#CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset bail --seed_num 5 --alpha 0.001 --beta 0.2

# ## best hyper-parameter for credit dataset
# echo '============Credit============='
# CUDA_VISIBLE_DEVICES=0 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset credit --seed_num 5 --alpha 0.5 --beta 0.1

# ## best hyper-parameter for pokec_z dataset
# echo '============Pokec_z============='
# CUDA_VISIBLE_DEVICES=1 python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset pokec_z --seed_num 5 --alpha 0.001 --beta 0.05

## best hyper-parameter for pokec_n dataset
# echo '============Pokec_n============='
# CUDA_VISIBLE_DEVICES=1 python train_copy.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset pokec_n --seed_num 5 --alpha 0.05 --beta 0.001

# 暴力循环大法号！尝试！
# for dataset in credit
# do
#     for lr in 1e-3 1e-4  # 0.01 0.1 0.5 1.0 2.0 3 5 10 15 20
#     do
#         for per in  0.3 0.5 0.7  # 2 5 10 15
#         do
#             for rs in 3 5 10 15 20  # 0 5 10 15 20 30 100 1000 5000
#             do
#                 #python train.py --dropout 0.5 --hidden 16 --lr=$lr --epochs 600 --model adagcn --dataset $dataset --seed_num 5 --alpha 0.5 --beta 0.001 --pre_train 0 --device 1 --avg True --rs=$rs --per=$per &
#                 python train.py --dropout 0.5 --hidden 16 --lr=$lr --epochs 600 --model adagcn --dataset $dataset --seed_num 5 --alpha 0.5 --beta 0.001 --pre_train 0 --device 1 --avg true --rs=$rs --per=$per --adv 1
#                 # 控制后台进程的数量，例如限制为4个
                
#                 done
#             done
#         done
       
#     done
# done
# for dataset in pokec_n
# do
#     for lr in 1e-3 1e-4
#     do
#         for epochs in 600 1000
#         do
#             for per in 0.3 0.5 0.7
#             do
#                 for rs in 3 5 10 15 20 30
#                 do
#                     # 依次顺序执行每个命令
#                     python train.py --dropout 0.5 --hidden 16 --lr=$lr --epochs=$epochs --model adagcn --dataset $dataset --seed_num 5 --alpha 0.05 --beta 0.001 --pre_train 0 --device 2 --avgy false --adv 0 --rs=$rs --per=$per 
#                 done
#             done
#         done
#     done
# done
# for dataset in income
# do
#     for lr in 1e-3 1e-4
#     do
#         for epochs in 600 1000
#         do
#             for per in 0.3 0.5 0.7
#             do
#                 for rs in 3 5 10 15 20 30
#                 do
#                     # 依次顺序执行每个命令
#                     python train.py --dropout 0.5 --hidden 16 --lr=$lr --epochs=$epochs --model adagcn --dataset $dataset --seed_num 5 --alpha 0.1 --beta 0.5 --pre_train 0 --device 2 --avgy true --adv 1 --rs=$rs --per=$per 
#                 done
#             done
#         done
#     done
# done
for dataset in pokec_z
do
    for lr in 1e-3 1e-4  # 0.01 0.1 0.5 1.0 2.0 3 5 10 15 20
    do
        for epochs in 400    # 2 5 10 15
        do
            for per in 0.3 0.5 0.7  # 2 5 10 15
            do
                for rs in 3 5 10 15 20 # 0 5 10 15 20 30 100 1000 5000
                do
                    python train.py --dropout 0.5 --hidden 16 --lr=$lr --epochs=$epochs --model adagcn --dataset $dataset --seed_num 5 --alpha 0.001 --beta 0.001 --pre_train 0 --device 2 --avgy true --adv 0 --rs=$rs --per=$per 
                done
            done
        done
    done
done




#python train_jq.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 600 --model adagcn --dataset credit --seed_num 5 --alpha 0.5 --beta 0.5 --pre_train 0 --device 1 --avg true --rs 5 --per 0.5 
#python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model adagcn --dataset pokec_n --seed_num 5 --alpha 0.05 --beta 0.05 --pre_train 0 --device 1 --avg True --rs 5 --per 0.3 
#python train_jq.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 600 --model adagcn --dataset bail --seed_num 5 --alpha 0.001 --beta 0.05 --pre_train 0 --device 1 --avg True --rs 5 --per 0.3


# pre_train
# python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 400 --model adagcn --dataset pokec_n --pre_seed 0 --alpha 0.05 --beta 0.001 --pre_train 1
# python train.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 400 --model adagcn --dataset income  --pre_seed 0 --alpha 0.1 --beta 0.2 --pre_train 1 --device 2
