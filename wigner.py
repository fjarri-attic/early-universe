import time
import numpy
import pickle
import os.path

from integrator import Integrator
from reikna.cluda import ocl_api, cuda_api, dtypes, Module, functions
from reikna.helpers import product


r_bohr = 5.2917720859e-11 # Bohr radius
hbar = 1.054571628e-34 # Planck constant
a_11 = 100. # scattering length
m_rb = 1.443160648e-25 # mass of the Rb atom, kg

dtype = numpy.complex128


def nonlinear_no_potential(dtype, U, nu):
    c_dtype = dtype
    c_ctype = dtypes.ctype(c_dtype)
    s_dtype = dtypes.real_for(dtype)
    s_ctype = dtypes.ctype(s_dtype)

    return Module.create(
        """
        %for comp in (0, 1):
        INLINE WITHIN_KERNEL ${c_ctype} ${prefix}${comp}(
            ${c_ctype} psi0, ${c_ctype} psi1, ${s_ctype} t)
        {
            return (
                ${mul}(psi${comp}, (
                    ${dtypes.c_constant(U[comp, 0])} * ${norm}(psi0) +
                    ${dtypes.c_constant(U[comp, 1])} * ${norm}(psi1)
                    ))
                - ${mul}(psi${1 - comp}, ${nu})
                );
        }
        %endfor
        """,
        render_kwds=dict(
            mul=functions.mul(c_dtype, s_dtype),
            norm=functions.norm(c_dtype),
            U=U,
            nu=dtypes.c_constant(nu, s_dtype),
            c_ctype=c_ctype,
            s_ctype=s_ctype))


def transpose_results(results):
    new_results = {key:[] for key in results[0].keys()}
    for res in results:
        for key in res:
            new_results[key].append(res[key])

    return new_results


def beam_splitter(psi):
    psi_c = psi.copy()
    psi_c[0] = (psi[0] + psi[1]) / numpy.sqrt(2)
    psi_c[1] = (psi[0] - psi[1]) / numpy.sqrt(2)
    return psi_c


class CollectorWigner:

    def __init__(self, ds):
        self.ds = ds

    def __call__(self, psi):
        psi_nobs = psi.get()
        psi = beam_splitter(psi_nobs)

        ns_nobs = numpy.abs(psi_nobs) ** 2 - 0.5 / product(self.ds)
        ns = numpy.abs(psi) ** 2 - 0.5 / product(self.ds)

        if len(psi.shape) == 3:
            # 1D
            n_nobs = ns_nobs.mean(1)
            n = ns.mean(1)
            Ns = (ns * self.ds[0]).sum(-1)
        elif len(psi.shape) == 4:
            # 2D
            n_nobs = ns_nobs.mean(1)
            n = ns.mean(1)
            Ns = (ns * product(self.ds)).sum(-1).sum(-1)
        elif len(psi.shape) == 5:
            # 3D
            n_nobs = (ns_nobs.mean(1) * self.ds[1] * self.ds[2]).sum(-1).sum(-1)
            n = (ns.mean(1) * self.ds[1] * self.ds[2]).sum(-1).sum(-1)
            Ns = (ns * product(self.ds)).sum(-1).sum(-1).sum(-1)

        res = dict(
            Nplus_mean=Ns[0].mean(), Nminus_mean=Ns[1].mean(),
            Nplus_std=Ns[0].std(), Nminus_std=Ns[1].std(),
            density=n,
            density_nobs=n_nobs)

        if len(psi.shape) == 5:
            res['slice_nobs'] = ns_nobs.mean(1)[:,:,:,ns.shape[-1] / 2]
            res['slice'] = ns.mean(1)[:,:,:,ns.shape[-1] / 2]

        return res


def run(dims, initial='same', gamma=1, nu=0, L_trap=10., samples=100,
        t=10, steps=5000, modes=512, ensembles=10, noise='coherent', T=0, U12=0):
    """
    Runs the simulation for

        i dpsi_j/dt = -nabla^2 psi_j / 2 + gamma |psi_j|^2 psi_j - nu psi_{1-j}

    With the initial state psi = 1 / sqrt(gamma)
    """
    assert initial in ('same', 'opposite')
    assert noise in ('coherent', 'bogolyubov')
    if noise == 'bogolyubov':
        assert dims == 1

    problem_shape = (modes,) * dims
    shape = (2, ensembles) + problem_shape
    box = (L_trap,) * dims
    ds = [L / points for L, points in zip(box, problem_shape)]
    fft_scale = numpy.sqrt(product(ds) / product(problem_shape))

    api = ocl_api()
    device = api.get_platforms()[0].get_devices()[1]
    thr = api.Thread(device)
    #thr = api.Thread.create()
    U = numpy.array([[gamma, U12 * gamma], [U12 * gamma, gamma]])
    integrator = Integrator(thr, shape, dtype, box, t, steps, samples,
        kinetic_coeff=0.5,
        nonlinear_module=nonlinear_no_potential(dtype, U, nu))

    psi = numpy.empty(shape, dtype)

    # Classical ground state

    psi.fill((1. / gamma) ** 0.5)

    if initial == 'opposite':
        psi[1] *= -1

    # To Wigner

    if noise == 'coherent':
        noise_kspace = numpy.sqrt(0.5) * (
            numpy.random.normal(size=shape, scale=numpy.sqrt(0.5))
            + 1j * numpy.random.normal(size=shape, scale=numpy.sqrt(0.5)))
    else:
        kgrid = numpy.fft.fftfreq(modes, L_trap / modes) * 2 * numpy.pi
        kgrid[0] = 1 # protection against warnings; we won't use this k anyway.

        # generate randoms
        Vk = T * (1. / (kgrid * numpy.sqrt(kgrid ** 2 + 2))) + 0.5
        Vk[0] = 0.5 # zeroth mode is just the vacuum mode
        betas = (
            numpy.random.normal(size=shape, scale=numpy.sqrt(0.5))
            + 1j * numpy.random.normal(size=shape, scale=numpy.sqrt(0.5))) * numpy.sqrt(Vk)

        phis = 0.5 * numpy.arctanh(1. / (kgrid ** 2 + 1))
        betas_m = betas.copy()
        betas_m[:,:,1:modes/2] = numpy.fliplr(betas[:,:,modes/2+1:])
        betas_m[:,:,modes/2+1:] = numpy.fliplr(betas[:,:,1:modes/2])
        noise_kspace = betas * numpy.cosh(phis) - betas_m.conj() * numpy.sinh(phis)
        noise_kspace[:,:,0] = betas[:,:,0] # zeroth mode is just the vacuum mode (no mixing)
        noise_kspace[:,:,modes/2] = betas[:,:,modes/2] # unmatched mode

    psi += numpy.fft.ifftn(noise_kspace, axes=range(2, len(shape))) / fft_scale
    psi = thr.to_device(psi)

    collector = CollectorWigner(ds)
    results, errors = integrator(psi, collector)
    results = transpose_results(results)

    results.update(dict(
        errors=errors,
        L_trap=L_trap, t=t, steps=steps, nu=nu, gamma=gamma,
        samples=samples, ensembles=ensembles))
    return results


def get(dim, **kwds):
    if not os.path.exists('results'):
        os.mkdir('results')

    name = "results/" + dim + " " + \
        " ".join([name + "=" + str(val) for name, val in kwds.items()]) + ".pickle"

    if os.path.exists(name):
        with open(name) as f:
            results = pickle.load(f)
    else:
        dims = {'1d': 1, '2d': 2, '3d': 3}
        results = run(dims[dim], **kwds)
        with open(name, 'w') as f:
            pickle.dump(results, f, protocol=2)

    return results
