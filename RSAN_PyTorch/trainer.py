import os
import sys
import torch

# Add base directory of references to path so that base_trainer can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from base_trainer import BaseTrainer


class RSANTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        dataset,
        optimizer,
        criterion,
        writer,
        val_dataset=None,
        patience=100,
        early_stopping_mode='max',
        early_stopping_metric='val_acc',
        snapshot_path='checkpoints',
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        device=None
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            criterion=criterion,
            writer=writer,
            val_dataset=val_dataset,
            patience=patience,
            early_stopping_mode=early_stopping_mode,
            early_stopping_metric=early_stopping_metric,
            snapshot_path=snapshot_path,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=device
        )
        self.last_train_acc = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, sampled_batch in enumerate(self.train_loader):
            inputs, targets, _ = self.process_batch(sampled_batch)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            train_acc += (preds == targets).float().mean().item() * inputs.size(0)
            
            self.iter_num += 1
            self.writer.add_scalar('train/loss_iter', loss.item(), self.iter_num)
            
        avg_train_loss = train_loss / len(self.train_dataset)
        avg_train_acc = train_acc / len(self.train_dataset)
        
        self.last_train_acc = avg_train_acc
        return avg_train_loss

    def val_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for sampled_batch in self.val_loader:
                inputs, targets, _ = self.process_batch(sampled_batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs > 0.5).float()
                val_acc += (preds == targets).float().mean().item() * inputs.size(0)
                
        avg_val_loss = val_loss / len(self.val_dataset)
        avg_val_acc = val_acc / len(self.val_dataset)
        
        # Save validation results to writer at every epoch
        self.writer.add_scalar('val/loss_epoch', avg_val_loss, epoch)
        self.writer.add_scalar('val/accuracy_epoch', avg_val_acc, epoch)
        
        return {
            'val_loss': avg_val_loss,
            'val_acc': avg_val_acc
        }

    def on_epoch_end(self, epoch, train_loss, val_result):
        val_loss = val_result['val_loss']
        val_acc = val_result['val_acc']
        # Print epoch results to console matching the original format
        print(f"Epoch {epoch + 1} - loss: {train_loss:.4f} - accuracy: {self.last_train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

    def log_checkpoint(self, epoch, filename):
        print(f"Saved checkpoint: {filename} at epoch {epoch + 1}")
