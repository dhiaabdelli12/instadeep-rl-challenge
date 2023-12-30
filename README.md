# instadeep-rl-challenge
This is my submission for the instadeep RL challenge in the second phase of the recruitement process.

## Configuration

#### Virtual environment setup
First, create a virutal environment. Make sure the `virtualenv` package is installed
```SHELL
python -m virtual .rlvenv
```
Then, activate the virtual environment
```SHELL
source .rlvenv/local/bin/activate
```
#### Installing dependencies
install the required dependencies found in `requirements.txt`
```SHELL
make install
```

## Training
To train the agent, run the following command:
```SHELL
make train
```

Training hyperparameters can be set in the `config.yml` file
```YAML
agent:
  epsilon_start: 1.0
  epsilon_end: .05
  epsilon_decay: 0.0005
  alpha: 0.001
  gamma: 0.99
  mem_size: 1000000
  batch_size: 64

training:
  eval_interval: 1
  plot_interval: 100
  n_episodes: 700
  checkpoint_interval: 100

evaluation:
  n_episodes: 20
```
The QNetwork checkpoints get saved periodically to `checkpoints/qnetwork` every `checkpoint_interval` number of iterations.
The saved checkpoint has the following structure:
```SHELL
qnetwork-{n_episodes}-{timestamp}.pth
```
## Monitoring
Reward and loss plots can be found in the `artefacts/` directory.

Real-time monitoring of the training can be done with Tensorboard by launching the command using `ctrl` + `shift` + `p` and typing "Launc Tensorboard" for VScode.


## Evaluation
After training, the agent can be evaluated on the environment while it visually renders with the following command:
```SHELL
make evaluate
```



