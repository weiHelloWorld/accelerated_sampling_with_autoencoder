# Accelerated sampling with data-augmented autoencoders

## Dependency

## Installation

## 1-min quick start

Let

`root_dir=accelerated_sampling_with_autoencoder/MD_simulation_on_alanine_dipeptide/current_work`

Go ahead to modify configuration file `${root_dir}/src/config.py`, and run 

```bash
python main_work.py
```

For more options, type

```bash
python main_work.py --help
```

## Quick introduction to data-augmented autoencoders

For traditional autoencoders, we minimize

$E=|A(x)-x|^2 + R$

where $A$ is autoencoder mapping function, $R$ is regularization term.

For data-augmented autoencoders, we minimize

$E=|A(x)-L(x)|^2 + R$

where $L$ is the alignment function responsible for data augmentation.

See slides for more information: (TODO)

## Directory structure

Directories are arranged as follows:

```
${root_dir}/src: source code
${root_dir}/target: output of simulation data (pdb files and coordinate files)
${root_dir}/resources: training results (autoencoders), and reference configurations files (pdb files)
${root_dir}/tests: test source code
```

## Testing

This package uses `nosetest` framework.  To run testing, run

```bash
cd ${root_dir}/tests
make test
```
