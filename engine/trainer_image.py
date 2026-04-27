import torch
import torch.amp as amp

from.trainer_base import BaseTrainer
from utils import timed
import time
import os
from config.constants import *
from models import ModelInput

class ImageFeatureTrainer(BaseTrainer):

    @timed
    def train(self):
        super(ImageFeatureTrainer, self).train()         
         

        for epoch in range(self.start_epoch+1, self.max_epochs+1):
            
            self.live_values.reset_metrics()

            self.scheduler.step(epoch)

            self.live_values.current_start_time = time.time()
            self.live_values.current_epoch = epoch

            self._run_epoch()
            self.on_epoch_end()
            
            if epoch % self.config.SOLVER.EVAL_PERIOD == 0 or epoch == 1:
                self.validation_step()
            if epoch % self.config.SOLVER.CHECKPOINT_PERIOD == 0:
                self.save_model(os.path.join(self.config.OUTPUT_DIR, self.config.MODEL.NAME + '_resume_{}.pth'.format(epoch)))

        self.on_training_end() 
                

    @timed
    def _run_epoch(self):
        self.model.train()
        for n_iter, batch in enumerate(self.train_loader):
            self.zero_grading()

            inputs = ModelInput(images=batch[IMG_INDEX].to(self.device),
                                labels=batch[PID_INDEX].to(self.device),
                                cam_ids=batch[CAMID_INDEX].to(self.device),
                                view_ids=batch[VIEWID_INDEX].to(self.device)
                                ) 

            if self.device == "cpu":
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs.labels)
                loss.backward()
                self.optimizer.step()
            else:
                with amp.autocast(self.device):
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, inputs.labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.live_values.update(loss, outputs, inputs.labels)
            self.log_training_details(n_iter)
            

    
        

    


        
        