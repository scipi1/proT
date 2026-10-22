"""
Optimization Forecaster with advanced multi-phase optimizer strategies.
Extends OnlineTargetForecaster with 7 different optimization modes.
"""

from typing import Any
import torch
import torch.nn as nn

from proT.training.forecasters.online_target_forecaster import OnlineTargetForecaster


class OptimizationForecaster(OnlineTargetForecaster):
    """
    Advanced forecaster with complex multi-phase optimization strategies.
    
    Extends OnlineTargetForecaster with:
    - Manual optimization control
    - 7 different optimizer configurations
    - Parameter splitting (embeddings vs model)
    - Mid-training optimizer switching
    - Learning rate schedulers with warmup
    - Optional gradient clipping
    
    This forecaster is designed for research experiments where fine-grained
    control over the optimization process is required.
    
    Args:
        config: Configuration dictionary containing:
            - All parameters from OnlineTargetForecaster
            - training.optimization: Integer 1-7 selecting optimizer mode
            - training.switch_epoch: Epoch to switch optimizers
            - training.switch_step: Step to switch schedulers
            - training.base_lr: Base learning rate for model parameters
            - training.emb_lr: Learning rate for embeddings (modes 2-7)
            - training.emb_start_lr: Initial LR for embeddings (modes 3-7)
            - training.warmup_steps: Warmup steps after switch (modes 5-7)
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Enable manual optimization
        self.automatic_optimization = False
        
        # Optimizer switching configuration
        self.switch_epoch = config["training"]["switch_epoch"]
        self.switch_step = config["training"]["switch_step"]
        
        # Optimizer switching state
        self._switched = False
        self.schedulers_flag = False
        
        # Validate optimization mode
        opt_mode = config["training"]["optimization"]
        assert opt_mode in [1, 2, 3, 4, 5, 6, 7], \
            f"Invalid optimization mode: {opt_mode}. Must be 1-7."
        
        print(f"✓ OptimizationForecaster initialized")
        print(f"  - Optimization mode: {opt_mode}")
        print(f"  - Manual optimization: enabled")
        print(f"  - Switch epoch: {self.switch_epoch}")
    
    def split_params(self):
        """
        Split model parameters into two groups:
        - Group 1: First 2 encoder embedding parameters
        - Group 2: All other parameters (model + decoder emb + remaining enc emb)
        
        This allows different learning rates for different parameter groups.
        
        Returns:
            Tuple of (group_1_params, group_2_params)
        """
        enc_emb_params = list(self.model.enc_embedding.embed_modules_list.parameters())
        dec_emb_params = list(self.model.dec_embedding.embed_modules_list.parameters())
        emb_param_ids = {id(p) for p in enc_emb_params + dec_emb_params}
        other_params = [p for p in self.model.parameters() if id(p) not in emb_param_ids]
        
        group_1 = enc_emb_params[:2]
        group_2 = other_params + dec_emb_params + enc_emb_params[2:]
        
        return group_1, group_2
    
    def training_step(self, batch, batch_idx):
        """
        Training step with manual optimization and optimizer switching.
        
        This overrides the parent training_step to implement:
        - Manual backward pass
        - Gradient clipping (modes 1-2)
        - Optimizer switching at switch_epoch
        - Separate optimizer steps for embeddings and model
        - Scheduler stepping
        """
        # Check if we should start showing targets (from parent)
        if self.current_epoch == self.epoch_show_trg:
            self.show_trg_active = True
        
        # Update target upper bound if in random mode (from parent)
        if self.current_epoch > self.epoch_show_trg and self.target_show_mode == "random":
            self._update_target_upper_bound()
        
        # Forward step
        loss, _, _ = self._step(batch=batch, stage="train")
        
        # Manual backward pass
        self.manual_backward(loss)
        
        # Gradient clipping for optimization modes 1 and 2
        if self.config["training"]["optimization"] in [1, 2]:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Get optimizers and schedulers
        opt_emb_now, opt_model_now, opt_emb_switch, opt_model_switch = self.optimizers()
        
        if self.lr_schedulers() is not None:
            model_scheduler_now, model_scheduler_switch = self.lr_schedulers()
            self.schedulers_flag = True
        
        # Check for optimizer switching
        if (not self._switched) and (self.current_epoch >= self.switch_epoch):
            # Switch to phase 2 optimizers
            self.optimizers()[0] = opt_emb_switch
            self.optimizers()[1] = opt_model_switch
            
            if self.schedulers_flag:
                self.lr_schedulers()[0] = model_scheduler_switch
            
            self._switched = True
            print(f"✓ Switched to phase 2 optimizers at epoch {self.current_epoch}")
        
        # Optimizer steps
        opt_emb_now.step()
        opt_model_now.step()
        
        # Scheduler step
        if self.schedulers_flag:
            model_scheduler_now.step()
        
        # Zero gradients
        opt_emb_now.zero_grad(set_to_none=True)
        opt_model_now.zero_grad(set_to_none=True)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizers based on the optimization mode.
        
        Returns 4 optimizers: [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        - p1: Phase 1 (before switch_epoch)
        - p2: Phase 2 (after switch_epoch)
        
        May also return schedulers for modes 5-7.
        
        Optimization Modes:
        1. Same AdamW for all parameters
        2. Adam with different LRs (embedding vs model)
        3. Adam + SparseAdam combination
        4. Adam + Adagrad combination
        5. SGD -> Adam switch with Adagrad embedding (with schedulers)
        6. SGD -> Adam switch with Adagrad embedding (different config, with schedulers)
        7. SGD -> Adam switch with Adagrad embedding (learn embedding then model, with schedulers)
        """
        # Helper function for creating schedulers
        def model_scheduler_p1(opt):
            return torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=5),
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=1E-4, total_iters=self.switch_step),
                ],
                milestones=[5]
            )
        
        def model_scheduler_p2(opt):
            return torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=1E-4, total_iters=self.switch_step),
                    torch.optim.lr_scheduler.LinearLR(opt, start_factor=1E-4, total_iters=self.config["training"]["warmup_steps"]),
                ],
                milestones=[self.switch_step]
            )
        
        # SGD kwargs
        SGD_kwargs = {"momentum": 0.0, "weight_decay": 0.0}
        
        # Optimization Mode 1: Same AdamW for all parameters
        if self.config["training"]["optimization"] == 1:
            opt_emb_p1 = torch.optim.AdamW(self.parameters(), lr=self.config["training"]["base_lr"])
            opt_emb_p2 = torch.optim.AdamW(self.parameters(), lr=self.config["training"]["base_lr"])
            opt_model_p1 = torch.optim.AdamW(self.parameters(), lr=self.config["training"]["base_lr"])
            opt_model_p2 = torch.optim.AdamW(self.parameters(), lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        # Optimization Mode 2: Adam with different LRs (embedding vs model)
        elif self.config["training"]["optimization"] == 2:
            group_1, group_2 = self.split_params()
            
            opt_emb_p1 = torch.optim.Adam(group_1, lr=self.config["training"]["emb_lr"])
            opt_emb_p2 = torch.optim.Adam(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        # Optimization Mode 3: Adam + SparseAdam combination
        elif self.config["training"]["optimization"] == 3:
            group_1, group_2 = self.split_params()
            
            opt_emb_p1 = torch.optim.SparseAdam(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.SparseAdam(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        # Optimization Mode 4: Adam + Adagrad combination
        elif self.config["training"]["optimization"] == 4:
            group_1, group_2 = self.split_params()
            
            opt_emb_p1 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_model_p1 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2]
        
        # Optimization Mode 5: SGD -> Adam switch with Adagrad embedding
        elif self.config["training"]["optimization"] == 5:
            group_1, group_2 = self.split_params()
            
            opt_emb_p1 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.SparseAdam(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1 = torch.optim.SGD(group_2, lr=self.config["training"]["base_lr"], **SGD_kwargs)
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            scheduler_model_p1 = {
                "scheduler": model_scheduler_p1(opt_model_p1),
                "interval": "epoch",
            }
            
            scheduler_model_p2 = {
                "scheduler": model_scheduler_p2(opt_model_p2),
                "interval": "epoch",
            }
            
            return [opt_emb_p1, opt_model_p1, opt_emb_p2, opt_model_p2], [scheduler_model_p1, scheduler_model_p2]
        
        # Optimization Mode 6: SGD -> Adam switch with Adagrad embedding (different config)
        elif self.config["training"]["optimization"] == 6:
            group_1, group_2 = self.split_params()
            
            opt_emb_p1 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1 = torch.optim.SGD(group_2, lr=self.config["training"]["base_lr"], **SGD_kwargs)
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            scheduler_model_p1 = {
                "scheduler": model_scheduler_p1(opt_model_p1),
                "interval": "epoch",
            }
            
            scheduler_model_p2 = {
                "scheduler": model_scheduler_p2(opt_model_p2),
                "interval": "epoch",
            }
            
            return [opt_emb_p1, opt_emb_p2, opt_model_p1, opt_model_p2], [scheduler_model_p1, scheduler_model_p2]
        
        # Optimization Mode 7: SGD -> Adam switch (learn embedding then model)
        elif self.config["training"]["optimization"] == 7:
            group_1, group_2 = self.split_params()
            
            opt_emb_p1 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_start_lr"])
            opt_emb_p2 = torch.optim.Adagrad(group_1, lr=self.config["training"]["emb_lr"])
            opt_model_p1 = torch.optim.SGD(group_2, lr=self.config["training"]["base_lr"], **SGD_kwargs)
            opt_model_p2 = torch.optim.Adam(group_2, lr=self.config["training"]["base_lr"])
            
            scheduler_model_p1 = {
                "scheduler": model_scheduler_p1(opt_model_p1),
                "interval": "epoch",
            }
            
            scheduler_model_p2 = {
                "scheduler": model_scheduler_p2(opt_model_p2),
                "interval": "epoch",
            }
            
            return [opt_emb_p1, opt_emb_p2, opt_model_p1, opt_model_p2], [scheduler_model_p1, scheduler_model_p2]
