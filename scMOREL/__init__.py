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
from scMORCEL.loss import ContrastiveLoss

try:
    from scMORCEL.score import combined_mahalanobis_energy
    ADVANCED_OOD_AVAILABLE = True
except ImportError:
    ADVANCED_OOD_AVAILABLE = False
    print("Advanced OOD method is not available.")

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
        validation_split=0.2,
        learning_rate=1e-3,
        score_function='mahalanobis_energy',  
        mahal_energy_alpha=0.7,
        energy_temperature=1.0,
        use_contrastive=True,
        contrastive_weight=0.1,
        contrastive_temperature=0.5,
        contrastive_type='contrastive',
        verbose=True
    ):
    
    if len(reference) != len(label):
        raise ValueError(
            f"The number of reference samples ({len(reference)}) does not match "
            f"the number of labels ({len(label)})."
        )

    
    if verbose:
        print("=" * 60)
        print("Starting data preprocessing")
        print("=" * 60)

    label = label.copy()
    label.columns = ['Label']
    
    status_dict = label['Label'].unique().tolist()
    int_label = label['Label'].apply(lambda x: status_dict.index(x))
    label.insert(1, 'transformed', int_label)
    
    X_train_full = reference.values
    X_test = test.values
    y_train_full = label['transformed'].values
    
    if use_validation:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=validation_split,
            random_state=42,
            stratify=y_train_full
        )
        
        if verbose:
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Test samples: {len(X_test)}")
    else:
        X_train = X_train_full
        y_train = y_train_full
        X_val = None
        y_val = None
        
        if verbose:
            if verbose:
            print(f"Training samples: {len(X_train)}")
            print("Validation set: None")
            print(f"Test samples: {len(X_test)}")
    
    dtype = torch.float
    X_train = torch.from_numpy(X_train).type(dtype)
    y_train = torch.from_numpy(y_train).type(torch.long)
    X_test = torch.from_numpy(X_test).type(dtype)
    
    if use_validation:
        X_val = torch.from_numpy(X_val).type(dtype)
        y_val = torch.from_numpy(y_val).type(torch.long)
        
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

    input_size = X_train.shape[1]
    num_class = len(status_dict)
    
    if verbose:
        print(f"\nModel configuration:")
        print(f"  - Input features: {input_size}")
        print(f"  - Number of classes: {num_class}")
        print(f"  - Model type: {model_type}")
    
    if model_type == 'basic':
        model = classifier(input_size, num_class)
    elif model_type == 'attention':
        model = AttentionClassifier(input_size, num_class, attention_heads=attention_heads)
        if verbose:
            print(f"  -  Using multi-head attention classifier (heads={attention_heads})")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    criterion = nn.CrossEntropyLoss()

    criterion_contrastive = None
    if use_contrastive:
        if contrastive_type == 'contrastive':
            criterion_contrastive = ContrastiveLoss(
                temperature=contrastive_temperature
            )
            if verbose:
                print("  - Supervised contrastive loss enabled")
                print(f"    - Weight: {contrastive_weight}")
                print(f"    - Temperature: {contrastive_temperature}")
        else:
            raise ValueError(f"Unknown contrastive loss type: {contrastive_type}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=max(5, patience // 2),
        verbose=verbose,
        min_lr=1e-6
    )

    if processing_unit == 'cpu':
        device = torch.device('cpu')
    elif processing_unit in ['gpu', 'cuda']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"  - Device: {device}")
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)
    
    model.to(device)

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
                
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                        print(
                            f"  - Validation loss did not improve for "
                            f"{patience} consecutive epochs."
                        )
                        print(f"  - Best validation loss: {best_val_loss:.4f}")
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
    
    if use_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nBest model loaded. Validation loss: {best_val_loss:.4f}")
    
    training_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed. Total time: {training_time:.2f} seconds")
        print("=" * 60)
    
    if verbose:
        print(f"\nStarting OOD scoring. Method: {score_function}")

    model.eval()

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
        print("OOD scoring completed.")
        print("\nScore statistics:")
        print(f"  - Min: {test_score.min():.4f}")
        print(f"  - Max: {test_score.max():.4f}")
        print(f"  - Mean: {test_score.mean():.4f}")
        print(f"  - Std: {test_score.std():.4f}")
        print("\nNote:")
        print("  - Higher scores indicate cells more likely to be known cells.")
        print("  - Lower scores indicate cells more likely to be rare cells.")

    return test_score, training_history
