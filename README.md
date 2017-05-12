# Accelerated sampling with data-augmented autoencoders

This is the framework for running accelerated sampling with data-augmented autoencoders.

## Dependency

OpenMM simulation pacakge: https://github.com/pandegroup/openmm

ANN_Force biasing force package: https://github.com/weiHelloWorld/ANN_Force

Keras: https://github.com/fchollet/keras

PyBrain (for backward compatibility): https://github.com/pybrain/pybrain

MDAnalysis: https://github.com/MDAnalysis/mdanalysis

Sklearn: https://github.com/scikit-learn/scikit-learn

Nose testing framework: https://github.com/nose-devs/nose

Some other Python scientific calculation packages are also needed, it is recommended to install them with Anaconda: https://www.continuum.io/downloads

For Linux/Ubuntu, you may use following script to install most of these packages:

```bash
echo "installing anaconda2 (please install it manually if you want the latest version)"
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
bash Anaconda2-4.3.1-Linux-x86_64.sh
export PATH="$HOME/.anaconda2/bin:$PATH"

echo "installing seaborn package"
conda install --channel https://conda.anaconda.org/anaconda seaborn

echo "installing setuptools"
conda install --channel https://conda.anaconda.org/anaconda setuptools

echo "installing MDAnalysis (including biopython)"
conda config --add channels MDAnalysis
conda install mdanalysis

echo "installing OpenMM"
conda install -c omnia openmm

echo "installing Keras"
pip install keras

echo "installing coverage (for nosetests)"
pip install coverage
```


## Installation and preparation

No installation is required.  You may simply have all dependent packages installed and checkout this repository.  Reference pdb files for simulation are needed.

It is **highly recommended** to run tests before running code to make sure packages are correctly installed.

## Testing

This package uses `nosetest` framework.  To run testing, run

```bash
root_dir=accelerated_sampling_with_autoencoder/MD_simulation_on_alanine_dipeptide/current_work
cd ${root_dir}/tests
make test
```

Tests include numerical unit tests (for tests with clear expected results) and figure plots (for others, such as training).

## 1-minute quick start

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

For multi-reference data-augmented autoencoders, error function is 

$E=\sum_j |A_j(x)-L_j(x)|^2 + R$

where $A_j$ are autoencoders that share all but the last layer, and $L_j$ is alignment functions with respect to reference $j$.

See slides for more information: (TODO)

## Directory structure

Directories are arranged as follows:

```
${root_dir}/src: source code
${root_dir}/target: output of simulation data (pdb files and coordinate files)
${root_dir}/resources: training results (autoencoders), and reference configurations files (pdb files)
${root_dir}/tests: test source code
```


## TODO

## Best practices

TODO

## Contact

For any questions, feel free to contact weichen9@illinois.edu or open a github issue.
