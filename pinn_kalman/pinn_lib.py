import os
import tensorboard
from pinn_kalman.pinn import PINN, B_PINN
import losses
import torch
import logging
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
import utils


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

    model = PINN(config)


    '''
    
    Training Schedule #1: Preliminary Training for FlowNet and PressureNet
    
    '''

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer_flow = losses.get_optimizer(config, model.flownet.parameters())
    optimizer_pres = losses.get_optimizer(config, model.pressurenet.parameters(), 0.001)
    state = dict(optimizer=(optimizer_flow, optimizer_pres), model=model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir, checkpoint_meta_dir = utils.get_ckptdir(workdir)
    # Resume training when intermediate checkpoints are detected
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(config,
                                             uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
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
            utils.save_checkpoint(checkpoint_meta_dir, state)
            print(f">>> temp checkpoint saved")

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            print(f">>> checkpoint_{save_step}.pth saved")


    '''

    Training Schedule #2: Regularization Training for PINN

    '''

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer_flow = losses.get_optimizer(config, model.flownet.parameters())
    optimizer_pres = losses.get_optimizer(config, model.pressurenet.parameters(), 0.005)
    state = dict(optimizer=(optimizer_flow, optimizer_pres), model=model, ema=ema, step=num_train_steps)

    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint_pinn.pth")
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

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
            utils.save_checkpoint(checkpoint_meta_dir, state)
            print(f">>> temp checkpoint saved")

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            print(f">>> checkpoint_{save_step}.pth saved")

def train_bpinn(config, workdir, ckpt_dir):

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    model = B_PINN(config)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer_flow = losses.get_optimizer(config, model.flownet.parameters(), is_bpinn=True)
    optimizer_pres = losses.get_optimizer(config, model.pressurenet.parameters(), is_bpinn=True, lr_mul=.05)
    state = dict(optimizer=(optimizer_flow, optimizer_pres), model=model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir, checkpoint_meta_dir = utils.get_ckptdir(workdir)
    # Resume training when intermediate checkpoints are detected
    state = utils.restore_bpinn_checkpoint(ckpt_dir, checkpoint_meta_dir, state, config)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(config,
                                             uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config, is_bpinn=True)
    train_step_fn = losses.get_prelim_step_fn(config, train=True, optimize_fn=optimize_fn, is_bpinn=True)
    eval_step_fn = losses.get_prelim_step_fn(config, train=False, optimize_fn=optimize_fn, is_bpinn=True)

    num_train_steps = config.training.n_bpinn_iters
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
            utils.save_checkpoint(checkpoint_meta_dir, state)
            print(f">>> temp checkpoint saved")

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            print(f">>> checkpoint_{save_step}.pth saved")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from configs.pinn.pinn_pde import get_config
    config = get_config()
    model = PINN(config)

        # Build data iterators
    _, eval_ds = datasets.get_dataset(config,
                                             uniform_dequantization=config.data.uniform_dequantization)
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

    workdir = "../workdir/pde-pinn/checkpoints-meta/checkpoint.pth"
    model = utils.load_checkpoint(workdir, model, config.device)
    f1, f2, x, y, t, target = unbatch(config, next(eval_iter))
    with torch.no_grad():
        flow_pred, pres_pred = model(f1, f2, x, y, t)

    workdir = "../workdir/pde-pinn/checkpoints-meta/checkpoint_pinn.pth"
    model = utils.load_checkpoint(workdir, model, config.device)
    with torch.no_grad():
        flow_pred_pinn, pres_pred_pinn = model(f1, f2, x, y, t)

    workdir = "../workdir/pde-bpinn/checkpoints-meta/checkpoint.pth"
    model = B_PINN(config)
    model = utils.load_checkpoint(workdir, model, config.device)
    with torch.no_grad():
        flow_pred_bpinn, pres_pred_bpinn, flow_std, pres_std = model.predict(f1, f2, x, y, t, n=64)


    mode = 0
    if mode == 0:

        fig = plt.figure(figsize=(30, 40))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 4, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])

        ax1.imshow(x[0, 0].cpu().detach().numpy())
        ax2.imshow(y[0, 0].cpu().detach().numpy())
        ax3.imshow(f2[0, 0].cpu().detach().numpy())

        u = torch.cat([target[0, 0], flow_pred[-1][0, 0], flow_pred_pinn[-1][0, 0], flow_pred_bpinn[0, 0]]).cpu().detach().numpy()
        v = torch.cat([target[0, 1], flow_pred[-1][0, 1], flow_pred_pinn[-1][0, 1], flow_pred_bpinn[0, 1]]).cpu().detach().numpy()
        p = torch.cat([target[0, 2], pres_pred[0, 0], pres_pred_pinn[0, 0], pres_pred_bpinn[0, 0]]).cpu().detach().numpy()

        ax4.imshow(u)
        ax5.imshow(v)
        ax6.imshow(p)

        ax7.imshow(flow_std[0, 0].cpu().detach().numpy())
        ax8.imshow(flow_std[0, 1].cpu().detach().numpy())
        ax9.imshow(pres_std[0, 0].cpu().detach().numpy())

        plt.show()

    elif mode == 1:
        from torchview import draw_graph

        model_graph = draw_graph(model, input_data=(f1, f2, x, y, t), device='cuda')
        model_graph.visual_graph

    else:

        mse_data = model.multiscale_data_mse(flow_pred, pres_pred, target)
        for extractor in model.model.module.feature_extractor.feature_extractors:
            for layer in extractor:
                if isinstance(layer, torch.nn.Conv2d):
                    grad = torch.autograd.grad(mse_data, layer.weight, retain_graph=True)
                    print(grad)
                    break


