import torch
import os
import logging
import time


class Clock:
    def __init__(self, itv):
        self.itv = itv
        self.start = time.time()

    def tic(self, info: str):
        elapsed_time = time.time() - self.start

        # Check if it's time to log
        if elapsed_time >= self.itv:
            print(info)
            self.start = time.time()  # Reset the timer


def show_memory_usage(device):
    print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")

def get_ckptdir(workdir, ckpt="checkpoints", ckpt_meta="checkpoints-meta"):
    checkpoint_dir = os.path.join(workdir, ckpt)
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, ckpt_meta, "checkpoint.pth")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    print(f"checkpoint_dir:{checkpoint_dir}")
    print(f"checkpoint_meta_dir:{checkpoint_meta_dir}")

    return checkpoint_dir, checkpoint_meta_dir


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)

        if loaded_state['info'] == 0:
            state['model'].load_state_dict(loaded_state['model'], strict=False)
            state['ema'].load_state_dict(loaded_state['ema'])
            state['step'] = loaded_state['step']
            state['optimizer'][0].load_state_dict(loaded_state['optimizer_1'])
            state['optimizer'][1].load_state_dict(loaded_state['optimizer_2'])
        else:
            state['optimizer'].load_state_dict(loaded_state['optimizer'])
            state['model'].load_state_dict(loaded_state['model'], strict=False)
            state['ema'].load_state_dict(loaded_state['ema'])
            state['step'] = loaded_state['step']

        return state

def restore_bpinn_checkpoint(pinn_ckpt_dir, bpinn_ckpt_dir, state, config):
    if not os.path.exists(bpinn_ckpt_dir):
        if not os.path.exists(pinn_ckpt_dir):
            logging.warning(f"None of checkpoints found at {bpinn_ckpt_dir} and {pinn_ckpt_dir}. "
                            f"Returned the same state as input")
            return state
        else:
            import pinn_kalman.pinn as pinn
            import losses
            pretrained = pinn.PINN(config)
            pretrained = load_checkpoint(pinn_ckpt_dir, pretrained, config.device)
            model = pinn.B_PINN(config, pretrained)

            optimizer_flow = losses.get_optimizer(config, model.flownet.parameters(), is_bpinn=True)
            optimizer_pres = losses.get_optimizer(config, model.pressurenet.parameters(), is_bpinn=True, lr_mul=.05)

            state['model'] = model
            state['optimizer'] = (optimizer_flow, optimizer_pres)

            logging.warning(f"PINN checkpoint loaded at {pinn_ckpt_dir}. "
                            f"Pretrained model returned")
            return state
    else:
        loaded_state = torch.load(bpinn_ckpt_dir, map_location=config.device)
        state['optimizer'][0].load_state_dict(loaded_state['optimizer_1'])
        state['optimizer'][1].load_state_dict(loaded_state['optimizer_2'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']

        return state





def load_checkpoint(ckpt_dir, model, device):
    if not os.path.exists(ckpt_dir):
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return model
    else:
        state = torch.load(ckpt_dir, map_location=device)["model"]
        model.load_state_dict(state)
        return model


def save_checkpoint(ckpt_dir, state):

    if isinstance(state['optimizer'], tuple):
        saved_state = {
            'info': 0,
            'optimizer_1': state['optimizer'][0].state_dict(),
            'optimizer_2': state['optimizer'][1].state_dict(),
            'model': state['model'].state_dict(),
            'ema': state['ema'].state_dict(),
            'step': state['step']
        }
    else:
        saved_state = {
            'info': 1,
            'optimizer': state['optimizer'].state_dict(),
            'model': state['model'].state_dict(),
            'ema': state['ema'].state_dict(),
            'step': state['step']
        }
    torch.save(saved_state, ckpt_dir)
