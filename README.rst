This is the simulation code for the paper `Quantum simulations of the early universe <http://arxiv.org/abs/1305.5314>`_. The files are:

* ``integrator.py``: contains the RK4IP stepper based on `reikna <http://reikna.publicfields.net>`_, and the corresponding integrator which performs full- and half-step propagation, comparing the results to estimate the convergence.
* ``wigner.py``: creates an initial state in Wigner representation and propagates it using the integrator.
* ``colors.xml``: a data file with values of main colors for plotting.
* ``mplhelpers.py``: backend settings for ``matplotlib`` and color/style definitions for quality plots.
* ``mencoder.sh``: a batch file used to create movies out of plots with ``mencoder``.
* ``plot.py``: the main file, plots the graphs for the paper.
