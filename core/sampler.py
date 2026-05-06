import json
import random
import numpy as np

from core.config import SAMPLER_STATE_FILE

# global sampler cache (per batch)

SAMPLERS = {}


# fast integer cyclic sampler (o(1), full coverage)

class CyclicIntegerSampler:
    def __init__(self, low, high, seed=None):
        self.low = int(low)
        self.high = int(high)
        self.range = self.high - self.low + 1
        self.k = 0

        if self.range <= 1:
            self.step = 1
        else:
            # pick step coprime with range
            self.step = random.choice(
                [s for s in range(1, self.range) if np.gcd(s, self.range) == 1]
            )

        self.offset = random.randint(0, self.range - 1)

    def next(self):
        val = self.low + (self.offset + self.k * self.step) % self.range
        self.k += 1
        return int(val)


def save_sampler_state():
    state = {}
    for key, sampler in SAMPLERS.items():
        state[key] = {
            "low": sampler.low,
            "high": sampler.high,
            "step": sampler.step,
            "offset": sampler.offset,
            "k": sampler.k
        }
    with open(SAMPLER_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_sampler_state():
    import os
    if not os.path.exists(SAMPLER_STATE_FILE):
        return
    with open(SAMPLER_STATE_FILE, "r") as f:
        state = json.load(f)

    for key, s in state.items():
        sampler = CyclicIntegerSampler(s["low"], s["high"])
        sampler.step = s["step"]
        sampler.offset = s["offset"]
        sampler.k = s["k"]
        SAMPLERS[key] = sampler
