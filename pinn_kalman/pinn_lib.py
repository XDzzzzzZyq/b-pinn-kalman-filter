import os
import tensorboard
from pinn_kalman.pinn import PINN_Net
import losses
import torch
import logging
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, load_checkpoint, restore_checkpoint


def unbatch(config, batch):
    f1, f2, coord_x, coord_y, t, target = batch
    return (f1.to(config.device).float(),
            f2.to(config.device).float(),
            coord_x.to(config.device).float().requires_grad_(),
            coord_y.to(config.device).float().requires_grad_(),
            t.to(config.device).float().requires_grad_(),
            target.to(config.device).float())

def train(config, workdir):

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    model = PINN_Net(config)


    '''
    
    Training Schedule #1: Preliminary Training for FlowNet and PressureNet
    
    '''

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer_flow = losses.get_optimizer(config, model.flownet.parameters())
    optimizer_pres = losses.get_optimizer(config, model.pressurenet.parameters(), 0.001)
    state = dict(optimizer=(optimizer_flow, optimizer_pres), model=model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    print(f"checkpoint_dir:{checkpoint_dir}")
    print(f"checkpoint_meta_dir:{checkpoint_meta_dir}")

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(config,
                                             uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    #continuous = config.training.continuous
    #reduce_mean = config.training.reduce_mean
    #likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_prelim_step_fn(config, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_prelim_step_fn(config, train=False, optimize_fn=optimize_fn)


    num_train_steps = config.training.n_iters
    print("num_train_steps", num_train_steps)

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting Preliminary Training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)

        # Execute one training step
        loss, v_loss, p_loss = train_step_fn(state, unbatch(config, batch))

        if step % config.training.log_freq == 0:
            logging.info(
                "step: %d, training_loss: %.5e = (%.5e, %.5e)" % (step, loss.item(), v_loss.item(), p_loss.item()))
            writer.add_scalar("training_vel_loss", v_loss, step)
            writer.add_scalar("training_prs_loss", p_loss, step)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)

                batch = next(eval_iter)

            loss, v_loss, p_loss = eval_step_fn(state, unbatch(config, batch))
            logging.info(
                "step: %d, eval_loss: %.5e = (%.5e, %.5e)" % (step, loss.item(), v_loss.item(), p_loss.item()))
            writer.add_scalar("eval_vel_loss", v_loss, step)
            writer.add_scalar("eval_prs_loss", p_loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)
            print(f">>> temp checkpoint saved")

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            print(f">>> checkpoint_{save_step}.pth saved")


    '''

    Training Schedule #2: Regularization Training for PINN

    '''

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer_pinn = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer_pinn, model=model, ema=ema, step=num_train_steps)

    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint_pinn.pth")
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    print(f"checkpoint_dir:{checkpoint_dir}")
    print(f"checkpoint_meta_dir:{checkpoint_meta_dir}")

    train_step_fn = losses.get_pinn_step_fn(config, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_pinn_step_fn(config, train=False, optimize_fn=optimize_fn)

    num_train_steps = config.training.n_iters + config.training.n_pinn_iters
    print("num_train_steps", num_train_steps)

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting Regularization Training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)
            batch = next(train_iter)

        # Execute one training step
        loss, loss_pinn, loss_data = train_step_fn(state, unbatch(config, batch))

        if step % config.training.log_freq == 0:
            logging.info(
                "step: %d, training_pinn_loss: %.5e = (%.5e, %.5e)" % (step, loss.item(), loss_pinn.item(), loss_data.item()))
            writer.add_scalar("training_pinn_loss", loss_pinn, step)
            writer.add_scalar("training_data_loss", loss_data, step)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)

                batch = next(eval_iter)

            loss, loss_pinn, loss_data = eval_step_fn(state, unbatch(config, batch))
            logging.info("step: %d, eval_pinn_loss: %.5e = (%.5e, %.5e)" % (
            step, loss.item(), loss_pinn.item(), loss_data.item()))
            writer.add_scalar("eval_pinn_loss", loss_pinn, step)
            writer.add_scalar("eval_data_loss", loss_data, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)
            print(f">>> temp checkpoint saved")

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            print(f">>> checkpoint_{save_step}.pth saved")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from configs.pinn.pinn_pde import get_config
    config = get_config()
    workdir = "../workdir/pde-pinn/checkpoints-meta/checkpoint.pth"

    model = PINN_Net(config)
    model = load_checkpoint(workdir, model, config.device)

        # Build data iterators
    _, eval_ds = datasets.get_dataset(config,
                                             uniform_dequantization=config.data.uniform_dequantization)
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    f1, f2, x, y, t, target = unbatch(config, next(eval_iter))
    veloc_pred, pressure_pred = model(f1, f2, x, y, t)

    mode = 0
    if mode == 0:

        fig, axe = plt.subplots(nrows=2, ncols=3, figsize=(40, 40))

        axe[0][0].imshow(x[0, 0].cpu().detach().numpy())
        axe[0][1].imshow(y[0, 0].cpu().detach().numpy())
        axe[0][2].imshow(f2[0, 0].cpu().detach().numpy())

        u = torch.cat([target[0, 0], veloc_pred[-1][0, 0]]).cpu().detach().numpy()
        v = torch.cat([target[0, 1], veloc_pred[-1][0, 1]]).cpu().detach().numpy()
        p = torch.cat([target[0, 2], pressure_pred[0, 0]]).cpu().detach().numpy()

        axe[1][0].imshow(u)
        axe[1][1].imshow(v)
        axe[1][2].imshow(p)

        plt.show()
        print(model.pressurenet.end[-1].weight, model.pressurenet.end[-1].bias)
        print(pressure_pred[0, 0].min(), pressure_pred[0, 0].max())
        print(t[0], target[0, 2].min(), target[0, 2].max())

    elif mode == 1:
        from torchview import draw_graph

        model_graph = draw_graph(model, input_data=(f1, f2, x, y, t), device='cuda')
        model_graph.visual_graph

    else:

        mse_data = model.multiscale_data_mse(veloc_pred, pressure_pred, target)
        for extractor in model.model.module.feature_extractor.feature_extractors:
            for layer in extractor:
                if isinstance(layer, torch.nn.Conv2d):
                    grad = torch.autograd.grad(mse_data, layer.weight, retain_graph=True)
                    print(grad)
                    break


