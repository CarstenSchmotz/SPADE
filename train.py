"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions  # Import TrainOptions for parsing command-line arguments
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import wandb

# Parse command-line options
opt = TrainOptions().parse()

# Initialize wandb
wandb.init(api_key='ef5596846579c28c73141548e0e374745de56bc3', project="spade_training")
wandb.config.update(opt)

# Print options to help debugging
print(' '.join(sys.argv))

# Load the dataset
dataloader = data.create_dataloader(opt)

# Create trainer for our model
trainer = Pix2PixTrainer(opt)

# Create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# Create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # Train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # Train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)
            
            # Log losses to wandb
            wandb.log({"epoch": epoch, "iteration": iter_counter.epoch_iter, **losses})

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)
            
            # Log images to wandb
            wandb.log({
                "input_label": [wandb.Image(visuals['input_label'], caption="Label")],
                "synthesized_image": [wandb.Image(visuals['synthesized_image'], caption="Synthesized Image")],
                "real_image": [wandb.Image(visuals['real_image'], caption="Real Image")]
            })

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
        
        # Log model checkpoint to wandb
        wandb.save('latest_net_G.pth')
        wandb.save('latest_net_D.pth')

print('Training was successfully finished.')
