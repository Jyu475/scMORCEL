import torch
import numpy as np
import torch.nn as nn
import time
import torch.utils.data as Data
from scMORCEL.classifier import classifier, AttentionClassifier
from scMORCEL.weightsampling import Weighted_Sampling
import copy
import pandas as pd
from sklearn.model_selection import train_test_split

# ========== 导入对比学习损失 ==========
from scMORCEL.loss import ContrastiveLoss

# ========== 尝试导入高级OOD方法 ==========
try:
    from scMORCEL.score import combined_mahalanobis_energy
    ADVANCED_OOD_AVAILABLE = True
except ImportError:
    ADVANCED_OOD_AVAILABLE = False
    print("高级OOD方法不可用")

def scMORCEL(
        test = None,
        reference = None, 
        label = None, 
        processing_unit = 'cuda',
        max_epochs=100,
        patience=10,
        model_type='attention',
        attention_heads=4,
        use_validation=True,
        validation_split=0.1,
        learning_rate=1e-3,
        
        # ========== 默认使用Mahalanobis+Energy ==========
        score_function='mahalanobis_energy',  # 修改默认值
        
        mahal_energy_alpha=0.5,
        energy_temperature=1.0,
        
        # 对比学习相关参数
        use_contrastive=True,
        contrastive_weight=0.1,
        contrastive_temperature=0.5,
        contrastive_type='contrastive',

        verbose=True
    ):
    
    if len(reference) != len(label):
        raise ValueError(
            f"训练集数据行数 {len(reference)} 与训练标签行数 {len(label)} 不匹配。"
        )

    
    # ========== 数据预处理 ==========
    if verbose:
        print("=" * 60)
        print("开始数据预处理")
        print("=" * 60)

    label = label.copy()
    label.columns = ['Label']
    
    status_dict = label['Label'].unique().tolist()
    int_label = label['Label'].apply(lambda x: status_dict.index(x))
    label.insert(1, 'transformed', int_label)
    
    X_train_full = reference.values
    X_test = test.values
    y_train_full = label['transformed'].values
    
    # ========== 划分训练集和验证集 ==========
    if use_validation:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=validation_split,
            random_state=42,
            stratify=y_train_full
        )
        
        if verbose:
            print(f"训练集样本数: {len(X_train)}")
            print(f"验证集样本数: {len(X_val)}")
            print(f"测试集样本数: {len(X_test)}")
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_val = None
        y_val = None
        
        if verbose:
            print(f"训练集样本数: {len(X_train)} (无验证集)")
            print(f"测试集样本数: {len(X_test)}")
    
    # 转换为 torch.Tensor
    dtype = torch.float
    X_train = torch.from_numpy(X_train).type(dtype)
    y_train = torch.from_numpy(y_train).type(torch.long)
    X_test = torch.from_numpy(X_test).type(dtype)
    
    if use_validation:
        X_val = torch.from_numpy(X_val).type(dtype)
        y_val = torch.from_numpy(y_val).type(torch.long)

    # ========== 构建 DataLoader ==========
    sampler = Weighted_Sampling(y_train)
    train_data = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        sampler=sampler,
        num_workers=1
    )
    
    if use_validation:
        val_data = Data.TensorDataset(X_val, y_val)
        val_loader = Data.DataLoader(
            dataset=val_data,
            batch_size=32,
            shuffle=False,
            num_workers=1
        )
    
    test_data = Data.TensorDataset(X_test, torch.zeros(len(X_test)))
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=32,
        num_workers=1
    )

    # ========== 模型初始化 ==========
    input_size = X_train.shape[1]
    num_class = len(status_dict)
    
    if verbose:
        print(f"\n模型配置:")
        print(f"  - 输入特征数: {input_size}")
        print(f"  - 类别数: {num_class}")
        print(f"  - 模型类型: {model_type}")
    
    if model_type == 'basic':
        model = classifier(input_size, num_class)
    elif model_type == 'attention':
        model = AttentionClassifier(input_size, num_class, attention_heads=attention_heads)
        if verbose:
            print(f"  - 使用多头注意力分类器 (heads={attention_heads})")
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    criterion = nn.CrossEntropyLoss()

    # 对比学习损失
    criterion_contrastive = None
    if use_contrastive:
        device_for_loss = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if contrastive_type == 'contrastive':
            criterion_contrastive = ContrastiveLoss(
                temperature=contrastive_temperature
            )
            if verbose:
                print(f"  ✓ 启用监督对比学习损失")
                print(f"    - 权重: {contrastive_weight}")
                print(f"    - 温度: {contrastive_temperature}")
        
        else:
            raise ValueError(f"未知的对比学习类型: {contrastive_type}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=max(5, patience // 2),
        verbose=verbose,
        min_lr=1e-6
    )

    # ========== 设备选择 ==========
    if processing_unit == 'cpu':
        device = torch.device('cpu')
    elif processing_unit in ['gpu', 'cuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"  - 计算设备: {device}")
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
    
    model.to(device)
    
     # ========== 训练循环 ==========
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    training_history = {
        'train_loss': [],
        'train_loss_ce': [],
        'train_loss_contrastive': [],
        'val_loss': [],
        'learning_rate': [],
        'epoch': []
    }
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_loss_ce = 0.0
        train_loss_contrastive = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss_ce = criterion(outputs, batch_y)
            
            loss_contrast = torch.tensor(0.0, device=device)
            if use_contrastive and hasattr(model, 'intermediate_features'):
                if model.intermediate_features is not None:
                    loss_contrast = criterion_contrastive(
                        model.intermediate_features,
                        batch_y
                    )
            
            loss = loss_ce + contrastive_weight * loss_contrast
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_ce += loss_ce.item()
            if use_contrastive:
                train_loss_contrastive += loss_contrast.item()
            train_batches += 1
        
        train_loss /= train_batches
        train_loss_ce /= train_batches
        train_loss_contrastive /= train_batches
        
        # 验证阶段
        if use_validation:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss_ce = criterion(outputs, batch_y)

                    loss_contrast = torch.tensor(0.0, device=device)
                    if use_contrastive and hasattr(model, 'intermediate_features'):
                        if model.intermediate_features is not None:
                           loss_contrast = criterion_contrastive(
                              model.intermediate_features,
                              batch_y
                            )

                    loss = loss_ce + contrastive_weight * loss_contrast

                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            training_history['train_loss'].append(train_loss)
            training_history['train_loss_ce'].append(train_loss_ce)
            training_history['train_loss_contrastive'].append(train_loss_contrastive)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rate'].append(current_lr)
            training_history['epoch'].append(epoch + 1)
            
            if verbose:
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Total: {train_loss:.4f} | "
                          f"Val: {val_loss:.4f} | "
                          f"CE: {train_loss_ce:.4f} | "
                          f"Contrast: {train_loss_contrastive:.4f} | "
                          f"LR: {current_lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"  ✓ 验证loss改善! 保存最佳模型")
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n早停触发 (epoch {epoch+1})")
                        print(f"  - 验证loss连续 {patience} 轮未改善")
                        print(f"  - 最佳验证loss: {best_val_loss:.4f}")
                    break
        else:
            training_history['train_loss'].append(train_loss)
            training_history['train_loss_ce'].append(train_loss_ce)
            training_history['train_loss_contrastive'].append(train_loss_contrastive)
            training_history['epoch'].append(epoch + 1)
            
            if verbose and (epoch + 1) % 10 == 0:
                if use_contrastive:
                    print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Total: {train_loss:.4f} | "
                          f"CE: {train_loss_ce:.4f} | "
                          f"Contrast: {train_loss_contrastive:.4f}")
                else:
                    print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                          f"Train: {train_loss:.4f}")
    
    # 加载最佳模型
    if use_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\n✓ 已加载最佳模型 (验证loss: {best_val_loss:.4f})")
    
    training_time = time.time() - start_time
    if verbose:
        print(f"\n训练完成! 总耗时: {training_time:.2f} 秒")
        print("=" * 60)
    
    # ========== OOD评分（只使用Mahalanobis+Energy）==========
    if verbose:
        print(f"\n开始OOD评分 (方法: {score_function})...")

    model.eval()

    
    # 使用Mahalanobis + Energy组合方法
    test_score, components = combined_mahalanobis_energy(
        train_loader,
        test_loader,
        model,
        device,
        alpha=mahal_energy_alpha,
        temperature=energy_temperature,
        verbose=verbose
    )

    if verbose:
        print("✓ OOD评分完成")
        print(f"\n评分统计:")
        print(f"  - 最小值: {test_score.min():.4f}")
        print(f"  - 最大值: {test_score.max():.4f}")
        print(f"  - 平均值: {test_score.mean():.4f}")
        print(f"  - 标准差: {test_score.std():.4f}")
        print(f"\n注意: 分数越高 → 越可能是正常细胞")
        print(f"      分数越低 → 越可能是稀有细胞")

    return test_score, training_history