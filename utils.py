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
