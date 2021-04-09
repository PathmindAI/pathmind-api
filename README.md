# Pathmind simulation Python API

An API to build multi-agent simulations in Python that can be run with Pathmind.
The interface is inspired by OpenAI gym, but has certain advantages:

- A Pathmind `Simulation` works for single and multiple agents.
- You only define reward terms, not a single reward value. You can either provide
your custom reward functions locally or craft them in the Pathmind app, without
having to go back to your original Python implementation.
- Similarly, you can provide as many useful observations as you like and later
select which ones to use for training in the Pathmind app.
- You can rely on Pathmind's scalable backend for training and don't have to
run your workloads on your machine.

## Installation

You can get the Pathmind API from PyPI:

```bash
pip install pathmind
```

## Usage

Simply implement the `Simulation` interface provided in `simulation.py`, see the
`examples` folder to get started.

## TODO

- [ ] better describe how things work (run through example, incl. obs selection etc.)
- [ ] explain how to run training on Pathmind
- [ ] explain how to use with Policy server
- [ ] open source this project
