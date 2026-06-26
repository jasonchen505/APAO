# APAO 项目 8卡3090 复现计划

## 一、资源评估

### 1.1 硬件资源
| 资源 | 规格 |
|------|------|
| GPU | 8 x RTX 3090 (24GB 显存) |
| 总显存 | 192GB |
| 适用场景 | 并行跑多个实验 / 加速单个实验 |

### 1.2 模型规模
| 模型 | 参数量 | 显存占用（估算） |
|------|--------|------------------|
| T5 (d_model=128, layers=4) | ~5M | < 2GB |
| Llama (hidden=128, layers=8) | ~10M | < 2GB |

**结论**：模型非常小，单卡3090绰绰有余，8卡可以并行跑多个实验。

### 1.3 数据集规模
| 数据集 | 用户数 | 物品数 | 交互数 | 平均序列长度 |
|--------|--------|--------|--------|--------------|
| Office | 4,905 | 2,420 | 53,258 | 10.9 |
| Grocery | 14,681 | 8,713 | 151,254 | 10.3 |
| Beauty | 22,363 | 12,101 | 198,502 | 8.9 |
| Yelp | 30,431 | 20,033 | 316,354 | 10.4 |

### 1.4 训练时间估算
| 配置 | 单数据集 | 全部4数据集 |
|------|----------|-------------|
| 单实验（200 epochs） | ~2-4小时 | ~8-16小时 |
| 超参搜索（30组合） | ~60-120小时 | ~240-480小时 |
| 并行8组 | ~8-15小时/数据集 | ~30-60小时 |

---

## 二、复现目标

### 2.1 核心目标
1. **复现论文 Table 1 的主要结果**（4个数据集 x 4种方法）
2. **理解 APAO 的核心机制**（前缀损失 + 自适应权重）
3. **掌握工程实现细节**（代码走读 + 调试）

### 2.2 方法清单
| 方法 | 类型 | 说明 |
|------|------|------|
| T5_CE | Baseline | 标准交叉熵 |
| T5_MSL | Baseline | Masked Softmax Loss |
| T5_APAO_Pointwise | Ours | 前缀点损失 |
| T5_APAO_Pairwise | Ours | 前缀对损失 |
| Llama_CE | Baseline | Decoder-only 版本 |
| Llama_APAO_Pointwise | Ours | Decoder-only 版本 |

---

## 三、复现计划

### Phase 0: 环境准备（Day 1 上午）

#### 0.1 创建环境
```bash
conda create -n apao python=3.12.11
conda activate apao

# PyTorch (CUDA 12.x)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 依赖
pip install -r requirements.txt
```

#### 0.2 验证环境
```bash
# 检查 GPU
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 检查 GCC 版本（需要 >= 9.0）
gcc --version
```

#### 0.3 配置 wandb（可选）
```bash
wandb login
# 或者禁用 wandb：export WANDB_MODE=disabled
```

---

### Phase 1: 单实验验证（Day 1 下午）

**目标**：跑通完整流程，理解代码

#### 1.1 跑通 CE Baseline（Office 数据集）
```bash
cd /home/chenyizhou/APAO
export PYTHONPATH=.:./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# T5_CE
python src/train/train_ce.py \
    --output_dir ./ckpt/debug_T5_CE \
    --seed 42 \
    --dataset Office \
    --method_name T5_CE

# 测试
python src/test/test.py \
    --ckpt_path ./ckpt/debug_T5_CE \
    --seed 42 \
    --dataset Office \
    --method_name T5_CE \
    --results_file ./results/debug_T5_CE \
    --test_batch_size 512 \
    --num_beams 20 \
    --is_constrained_beam_search
```

#### 1.2 跑通 APAO Pointwise（Office 数据集）
```bash
python src/train/train_apao.py \
    --output_dir ./ckpt/debug_T5_APAO_Pointwise \
    --seed 42 \
    --dataset Office \
    --method_name T5_APAO_Pointwise \
    --beta 0.3 \
    --eta_adapt 1e-4

python src/test/test.py \
    --ckpt_path ./ckpt/debug_T5_APAO_Pointwise \
    --seed 42 \
    --dataset Office \
    --method_name T5_APAO_Pointwise \
    --results_file ./results/debug_T5_APAO_Pointwise \
    --test_batch_size 512 \
    --num_beams 20 \
    --is_constrained_beam_search
```

#### 1.3 检查结果
- 训练日志是否正常
- 损失是否下降
- 测试指标是否合理

---

### Phase 2: 并行复现 Baseline（Day 2）

**目标**：用 8 卡并行跑完所有 Baseline

#### 2.1 并行策略
```
GPU 0: T5_CE on Office
GPU 1: T5_CE on Grocery
GPU 2: T5_CE on Beauty
GPU 3: T5_CE on Yelp
GPU 4: Llama_CE on Office
GPU 5: Llama_CE on Grocery
GPU 6: Llama_CE on Beauty
GPU 7: Llama_CE on Yelp
```

#### 2.2 并行脚本
```bash
#!/bin/bash
# run_parallel_baselines.sh

export PYTHONPATH=.:./src:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASETS=(Office Grocery Beauty Yelp)
METHODS=(T5_CE Llama_CE)
SEED=42
beam_size=20
test_bsz=512

GPU_ID=0

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        export WANDB_PROJECT="${dataset}_${method}"
        export WANDB_RUN_NAME="${method}_seed${SEED}"
        OUTPUT_DIR=./ckpt/${dataset}_${method}/${WANDB_RUN_NAME}
        RESULTS_FILE=./results/${dataset}_${method}/${WANDB_RUN_NAME}
        
        # 后台运行
        (
            python src/train/train_ce.py \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --dataset $dataset \
                --method_name $method
            
            python src/test/test.py \
                --ckpt_path $OUTPUT_DIR \
                --seed $SEED \
                --dataset $dataset \
                --method_name $method \
                --results_file $RESULTS_FILE \
                --test_batch_size $test_bsz \
                --num_beams $beam_size \
                --is_constrained_beam_search
        ) > ./logs/${dataset}_${method}.log 2>&1 &
        
        GPU_ID=$((GPU_ID + 1))
        if [ $GPU_ID -ge 8 ]; then
            GPU_ID=0
            wait  # 等待当前批次完成
        fi
    done
done

wait
echo "All baselines completed!"
```

---

### Phase 3: 超参搜索 - APAO Pointwise（Day 3-4）

**目标**：搜索最佳 β 和 η

#### 3.1 搜索空间
| 超参数 | 搜索范围 |
|--------|----------|
| β | [0.05, 0.1, 0.2, 0.3, 0.4] |
| η_adapt | [5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 5e-4] |

总组合：5 x 6 = 30 组

#### 3.2 并行策略
- 8 卡并行，每卡跑一组
- 分 4 批跑完 30 组（8+8+8+6）

#### 3.3 超参搜索脚本
```bash
#!/bin/bash
# run_hp_search_pointwise.sh

export PYTHONPATH=.:./src:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET=Office  # 先在 Office 上搜索
method_name=T5_APAO_Pointwise
SEED=42
beam_size=20
test_bsz=512

beta_values=(0.05 0.1 0.2 0.3 0.4)
eta_values=(5e-6 1e-5 3e-5 5e-5 1e-4 5e-4)

GPU_ID=0
JOB_COUNT=0

for beta in "${beta_values[@]}"; do
    for eta in "${eta_values[@]}"; do
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        export WANDB_PROJECT="${DATASET}_${method_name}"
        export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_eta${eta}"
        OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}
        RESULTS_FILE=./results/${DATASET}_${method_name}/${WANDB_RUN_NAME}
        
        (
            python src/train/train_apao.py \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --dataset $DATASET \
                --method_name $method_name \
                --beta $beta \
                --eta_adapt $eta
            
            python src/test/test.py \
                --ckpt_path $OUTPUT_DIR \
                --seed $SEED \
                --dataset $DATASET \
                --method_name $method_name \
                --results_file $RESULTS_FILE \
                --test_batch_size $test_bsz \
                --num_beams $beam_size \
                --is_constrained_beam_search
        ) > ./logs/${DATASET}_${method_name}_b${beta}_e${eta}.log 2>&1 &
        
        GPU_ID=$((GPU_ID + 1))
        JOB_COUNT=$((JOB_COUNT + 1))
        
        if [ $GPU_ID -ge 8 ]; then
            GPU_ID=0
            wait
        fi
    done
done

wait
echo "HP search completed!"
```

#### 3.4 结果收集
```python
# collect_results.py
import json
import os
import glob

results = {}
for result_file in glob.glob("./results/Office_T5_APAO_Pointwise/*/final_beam_search_results.json"):
    with open(result_file, 'r') as f:
        data = json.load(f)
    run_name = os.path.basename(os.path.dirname(result_file))
    results[run_name] = data['final_results']

# 按 NDCG@10 排序
sorted_results = sorted(results.items(), key=lambda x: x[1].get('ndcg@10', 0), reverse=True)
for name, metrics in sorted_results[:5]:
    print(f"{name}: {metrics}")
```

---

### Phase 4: 超参搜索 - APAO Pairwise（Day 5-6）

**目标**：搜索 APAO Pairwise 的最佳超参数

#### 4.1 额外超参数
| 超参数 | 搜索范围 |
|--------|----------|
| β | [0.05, 0.1, 0.2, 0.3, 0.4] |
| η_adapt | [5e-6, 1e-5, 3e-5, 5e-5, 1e-4, 5e-4] |
| neg_k | [50, 100]（可选） |

#### 4.2 并行策略
同 Phase 3

---

### Phase 5: 最终复现（Day 7-8）

**目标**：用最佳超参数跑完所有数据集

#### 5.1 最佳超参数（根据论文）
| 数据集 | T5_APAO_Pointwise | T5_APAO_Pairwise |
|--------|-------------------|------------------|
| Office | β=0.3, η=1e-4 | β=0.2, η=3e-5 |
| Grocery | β=0.1, η=5e-5 | β=0.1, η=5e-5 |
| Beauty | β=0.1, η=1e-5 | β=0.1, η=1e-4 |
| Yelp | β=0.05, η=1e-5 | β=0.05, η=5e-6 |

#### 5.2 完整复现脚本
```bash
#!/bin/bash
# run_full_reproduction.sh

export PYTHONPATH=.:./src:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SEED=42
beam_size=20
test_bsz=512

# 方法和数据集配置
DATASET_values=(Office Grocery Beauty Yelp Office Grocery Beauty Yelp)
method_name_values=(T5_APAO_Pointwise T5_APAO_Pointwise T5_APAO_Pointwise T5_APAO_Pointwise T5_APAO_Pairwise T5_APAO_Pairwise T5_APAO_Pairwise T5_APAO_Pairwise)
beta_values=(0.3 0.1 0.1 0.05 0.2 0.1 0.1 0.05)
eta_values=(1e-4 5e-5 1e-5 1e-5 3e-5 5e-5 1e-4 5e-6)

GPU_ID=0

for i in "${!DATASET_values[@]}"; do
    DATASET=${DATASET_values[i]}
    method_name=${method_name_values[i]}
    beta=${beta_values[i]}
    eta=${eta_values[i]}
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    export WANDB_PROJECT="${DATASET}_${method_name}"
    export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_eta${eta}"
    OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}
    RESULTS_FILE=./results/${DATASET}_${method_name}/${WANDB_RUN_NAME}
    
    (
        python src/train/train_apao.py \
            --output_dir $OUTPUT_DIR \
            --seed $SEED \
            --dataset $DATASET \
            --method_name $method_name \
            --beta $beta \
            --eta_adapt $eta
        
        python src/test/test.py \
            --ckpt_path $OUTPUT_DIR \
            --seed $SEED \
            --dataset $DATASET \
            --method_name $method_name \
            --results_file $RESULTS_FILE \
            --test_batch_size $test_bsz \
            --num_beams $beam_size \
            --is_constrained_beam_search
    ) > ./logs/${DATASET}_${method_name}_final.log 2>&1 &
    
    GPU_ID=$((GPU_ID + 1))
    if [ $GPU_ID -ge 8 ]; then
        GPU_ID=0
        wait
    fi
done

wait
echo "Full reproduction completed!"
```

---

## 四、学习计划

### Day 1: 环境 + 代码走读
- [ ] 环境搭建
- [ ] 跑通单个实验
- [ ] 代码走读：数据加载 (`utils/data.py`)
- [ ] 代码走读：模型结构 (`src/models/`)

### Day 2: Baseline 复现 + 理解
- [ ] 并行跑完 Baseline
- [ ] 理解 CE 和 MSL 的区别
- [ ] 理解约束 Beam Search

### Day 3-4: APAO Pointwise 深入
- [ ] 超参搜索
- [ ] 理解前缀损失计算
- [ ] 理解自适应权重机制
- [ ] 分析超参对性能的影响

### Day 5-6: APAO Pairwise 深入
- [ ] 超参搜索
- [ ] 理解对比学习损失
- [ ] 理解前缀掩码机制
- [ ] 分析 Pointwise vs Pairwise

### Day 7-8: 最终复现 + 总结
- [ ] 用最佳超参数跑完所有实验
- [ ] 对比论文结果
- [ ] 总结学习心得

---

## 五、关键检查点

### 5.1 训练监控
```bash
# 查看训练日志
tail -f ./logs/Office_T5_CE.log

# 查看 GPU 使用
watch -n 1 nvidia-smi
```

### 5.2 结果验证
| 方法 | Office NDCG@10 | Grocery NDCG@10 | Beauty NDCG@10 | Yelp NDCG@10 |
|------|----------------|-----------------|----------------|--------------|
| T5_CE (论文) | ~0.05 | ~0.04 | ~0.03 | ~0.02 |
| T5_APAO_Pointwise (论文) | ~0.07 | ~0.06 | ~0.05 | ~0.04 |

（注：具体数值需要看论文 Table 1）

### 5.3 常见问题
1. **OOM**：减小 batch_size
2. **训练太慢**：检查是否用了 GPU
3. **指标为 0**：检查 tokenizer 和数据加载

---

## 六、时间线总结

| 阶段 | 时间 | 任务 | GPU 使用 |
|------|------|------|----------|
| Phase 0 | Day 1 上午 | 环境准备 | - |
| Phase 1 | Day 1 下午 | 单实验验证 | 1卡 |
| Phase 2 | Day 2 | Baseline 复现 | 8卡并行 |
| Phase 3 | Day 3-4 | Pointwise 超参搜索 | 8卡并行 |
| Phase 4 | Day 5-6 | Pairwise 超参搜索 | 8卡并行 |
| Phase 5 | Day 7-8 | 最终复现 | 8卡并行 |

**总计**：约 8 个工作日完成全部复现

---

## 七、进阶探索（可选）

### 7.1 Llama 版本复现
- 复现 Llama_CE 和 Llama_APAO_Pointwise
- 对比 T5 和 Llama 的效果

### 7.2 消融实验
- 去掉前缀损失
- 去掉自适应权重
- 不同前缀长度的影响

### 7.3 改进尝试
- 难负例挖掘
- 课程学习
- 不同编码方式

---

## 附录：快速命令参考

### 环境相关
```bash
# 创建环境
conda create -n apao python=3.12.11 && conda activate apao

# 安装依赖
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### 训练相关
```bash
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python src/train/train_apao.py --method_name T5_APAO_Pointwise --dataset Office --beta 0.3 --eta_adapt 1e-4

# 查看日志
tail -f ./logs/xxx.log

# 查看 GPU
watch -n 1 nvidia-smi
```

### 测试相关
```bash
# 测试
python src/test/test.py --ckpt_path ./ckpt/xxx --dataset Office --method_name T5_APAO_Pointwise --is_constrained_beam_search
```
