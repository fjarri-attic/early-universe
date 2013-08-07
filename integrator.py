from __future__ import print_function

import numpy
import sys
import time

from reikna.cluda import dtypes, Module, Snippet, functions
from reikna.core import Computation, Parameter, Annotation, Transformation, Type

from reikna.fft import FFT
from reikna.pureparallel import PureParallel


def get_ksquared(shape, box):
    ks = [
        2 * numpy.pi * numpy.fft.fftfreq(size, length / size)
        for size, length in zip(shape, box)]

    if len(shape) > 1:
        full_ks = numpy.meshgrid(*ks, indexing='ij')
    else:
        full_ks = [ks]

    return sum([full_k ** 2 for full_k in full_ks])


def get_nonlinear_wrapper(c_dtype, nonlinear_module, dt):
    s_dtype = dtypes.real_for(c_dtype)
    return Module.create(
        """
        %for comp in (0, 1):
        INLINE WITHIN_KERNEL ${c_ctype} ${prefix}${comp}(
            ${c_ctype} psi0, ${c_ctype} psi1, ${s_ctype} t)
        {
            ${c_ctype} nonlinear = ${nonlinear}${comp}(psi0, psi1, t);
            return ${mul}(
                COMPLEX_CTR(${c_ctype})(0, -${dt}),
                nonlinear);
        }
        %endfor
        """,
        render_kwds=dict(
            c_ctype=dtypes.ctype(c_dtype),
            s_ctype=dtypes.ctype(s_dtype),
            mul=functions.mul(c_dtype, c_dtype),
            dt=dtypes.c_constant(dt, s_dtype),
            nonlinear=nonlinear_module))


def get_nonlinear1(state_arr, scalar_dtype, nonlinear_module):
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>
        ${output.ctype} psi0 = ${input.load_idx}(0, ${all_indices});
        ${output.ctype} psi1 = ${input.load_idx}(1, ${all_indices});

        ${output.store_idx}(0, ${all_indices}, ${nonlinear}0(psi0, psi1, ${t}));
        ${output.store_idx}(1, ${all_indices}, ${nonlinear}1(psi0, psi1, ${t}));
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(nonlinear=nonlinear_module))


def get_nonlinear2(state_arr, scalar_dtype, nonlinear_module):
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('p', Annotation(state_arr, 'i')),
            Parameter('pk', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>

        ${output.ctype} p0 = ${p.load_idx}(0, ${all_indices});
        ${output.ctype} p1 = ${p.load_idx}(1, ${all_indices});
        ${output.ctype} pk0 = ${pk.load_idx}(0, ${all_indices});
        ${output.ctype} pk1 = ${pk.load_idx}(1, ${all_indices});

        ${output.ctype} t1_0 = ${nonlinear}0(${div}(pk0, 2) + p0, ${div}(pk1, 2) + p1, ${t});
        ${output.ctype} t1_1 = ${nonlinear}1(${div}(pk0, 2) + p0, ${div}(pk1, 2) + p1, ${t});

        ${output.ctype} t2_0 = ${div}(pk0, 6) + p0;
        ${output.ctype} t2_1 = ${div}(pk1, 6) + p1;

        ${output.ctype} t3_0 = ${nonlinear}0(${div}(t1_0, 2) + p0, ${div}(t1_1, 2) + p1, ${t});
        ${output.ctype} t3_1 = ${nonlinear}1(${div}(t1_0, 2) + p0, ${div}(t1_1, 2) + p1, ${t});

        ${output.store_idx}(0, ${all_indices}, ${div}(t1_0, 3) + t2_0 + ${div}(t3_0, 3));
        ${output.store_idx}(1, ${all_indices}, ${div}(t1_1, 3) + t2_1 + ${div}(t3_1, 3));
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            nonlinear=nonlinear_module,
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


def get_nonlinear3(state_arr, scalar_dtype, nonlinear_module):
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('p', Annotation(state_arr, 'i')),
            Parameter('pk', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>

        ${output.ctype} p0 = ${p.load_idx}(0, ${all_indices});
        ${output.ctype} p1 = ${p.load_idx}(1, ${all_indices});
        ${output.ctype} pk0 = ${pk.load_idx}(0, ${all_indices});
        ${output.ctype} pk1 = ${pk.load_idx}(1, ${all_indices});

        ${output.ctype} t1_0 = ${nonlinear}0(${div}(pk0, 2) + p0, ${div}(pk1, 2) + p1, ${t});
        ${output.ctype} t1_1 = ${nonlinear}1(${div}(pk0, 2) + p0, ${div}(pk1, 2) + p1, ${t});

        ${output.ctype} t2_0 = ${nonlinear}0(${div}(t1_0, 2) + p0, ${div}(t1_1, 2) + p1, ${t});
        ${output.ctype} t2_1 = ${nonlinear}1(${div}(t1_0, 2) + p0, ${div}(t1_1, 2) + p1, ${t});

        ${output.store_idx}(0, ${all_indices}, t2_0 + p0);
        ${output.store_idx}(1, ${all_indices}, t2_1 + p1);
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            nonlinear=nonlinear_module,
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


def get_combine(state_arr, scalar_dtype, nonlinear_module):
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('p2', Annotation(state_arr, 'i')),
            Parameter('p3', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))],
        """
        <%
            all_indices = ', '.join(idxs)
        %>

        ${output.ctype} p2_0 = ${p2.load_idx}(0, ${all_indices});
        ${output.ctype} p2_1 = ${p2.load_idx}(1, ${all_indices});
        ${output.ctype} p3_0 = ${p3.load_idx}(0, ${all_indices});
        ${output.ctype} p3_1 = ${p3.load_idx}(1, ${all_indices});

        ${output.store_idx}(0, ${all_indices}, p2_0 + ${div}(${nonlinear}0(p3_0, p3_1, ${t}), 6));
        ${output.store_idx}(1, ${all_indices}, p2_1 + ${div}(${nonlinear}1(p3_0, p3_1, ${t}), 6));
        """,
        guiding_array=state_arr.shape[1:],
        render_kwds=dict(
            nonlinear=nonlinear_module,
            div=functions.div(state_arr.dtype, numpy.int32, out_dtype=state_arr.dtype)))


class RK4IPStepper(Computation):

    def __init__(self, state_arr, dt, box=None, kinetic_coeff=1, nonlinear_module=None):
        scalar_dtype = dtypes.real_for(state_arr.dtype)
        Computation.__init__(self, [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('t', Annotation(scalar_dtype))])

        self._box = box
        self._kinetic_coeff = kinetic_coeff
        self._nonlinear_module = nonlinear_module
        self._components = state_arr.shape[0]
        self._ensembles = state_arr.shape[1]
        self._grid_shape = state_arr.shape[2:]

        ksquared = get_ksquared(self._grid_shape, self._box)
        self._kprop = numpy.exp(ksquared * (-1j * kinetic_coeff * dt / 2)).astype(state_arr.dtype)
        self._kprop_trf = Transformation(
            [
                Parameter('output', Annotation(state_arr, 'o')),
                Parameter('input', Annotation(state_arr, 'i')),
                Parameter('kprop', Annotation(self._kprop, 'i'))],
            """
            ${kprop.ctype} kprop_coeff = ${kprop.load_idx}(${', '.join(idxs[2:])});
            ${output.store_same}(${mul}(${input.load_same}, kprop_coeff));
            """,
            render_kwds=dict(mul=functions.mul(state_arr.dtype, self._kprop.dtype)))

        self._fft = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop = FFT(state_arr, axes=range(2, len(state_arr.shape)))
        self._fft_with_kprop.parameter.output.connect(
            self._kprop_trf, self._kprop_trf.input,
            output_prime=self._kprop_trf.output,
            kprop=self._kprop_trf.kprop)

        nonlinear_wrapper = get_nonlinear_wrapper(state_arr.dtype, nonlinear_module, dt)
        self._N1 = get_nonlinear1(state_arr, scalar_dtype, nonlinear_wrapper)
        self._N2 = get_nonlinear2(state_arr, scalar_dtype, nonlinear_wrapper)
        self._N3 = get_nonlinear3(state_arr, scalar_dtype, nonlinear_wrapper)
        self._combine = get_combine(state_arr, scalar_dtype, nonlinear_wrapper)

    def _add_kprop(self, plan, output, input_, kprop_device):
        temp = plan.temp_array_like(output)
        plan.computation_call(self._fft_with_kprop, temp, kprop_device, input_)
        plan.computation_call(self._fft, output, temp, inverse=True)

    def _build_plan(self, plan_factory, device_params, output, input_, t):

        plan = plan_factory()

        kprop_device = plan.persistent_array(self._kprop)

        p = plan.temp_array_like(output)
        self._add_kprop(plan, p, input_, kprop_device)

        temp = plan.temp_array_like(output)
        pk = plan.temp_array_like(output)
        plan.computation_call(self._N1, temp, input_, t)
        self._add_kprop(plan, pk, temp, kprop_device)

        temp = plan.temp_array_like(output)
        p2 = plan.temp_array_like(output)
        plan.computation_call(self._N2, temp, p, pk, t)
        self._add_kprop(plan, p2, temp, kprop_device)

        temp = plan.temp_array_like(output)
        p3 = plan.temp_array_like(output)
        plan.computation_call(self._N3, temp, p, pk, t)
        self._add_kprop(plan, p3, temp, kprop_device)

        plan.computation_call(self._combine, output, p2, p3, t)

        return plan


class Integrator:

    def __init__(self, thr, shape, dtype, box, tmax, steps, samples,
            kinetic_coeff=1, nonlinear_module=None):

        state_arr = Type(dtype, shape)
        self.tmax = tmax
        self.steps = steps
        self.samples = samples
        self.dt = float(tmax) / steps
        self.dt_half = self.dt / 2

        self.thr = thr
        self.stepper = RK4IPStepper(state_arr, self.dt,
            box=box, kinetic_coeff=kinetic_coeff, nonlinear_module=nonlinear_module).compile(thr)
        self.stepper_half = RK4IPStepper(state_arr, self.dt_half,
            box=box, kinetic_coeff=kinetic_coeff, nonlinear_module=nonlinear_module).compile(thr)

    def _integrate(self, psi, half_step, collector):
        results = []

        t_collectors = 0

        t_start = time.time()

        if half_step:
            t_collector = time.time()
            results.append(collector(psi))
            t_collectors += time.time() - t_collector

        stepper = self.stepper_half if half_step else self.stepper
        dt = self.dt_half if half_step else self.dt
        step = 0
        sample = 0
        t = 0

        if half_step:
            print("Sampling at t =", end=' ')
        else:
            print("Skipping callbacks at t =", end=' ')

        for step in xrange(self.steps * (2 if half_step else 1)):
            stepper(psi, psi, t)
            t += dt

            if (step + 1) % (self.steps / self.samples * (2 if half_step else 1)) == 0:
                if half_step:
                    print(t, end=' ')
                    sys.stdout.flush()
                    t_collector = time.time()
                    results.append(collector(psi))
                    t_collectors += time.time() - t_collector
                else:
                    print(t, end=' ')
                    sys.stdout.flush()

        print()

        t_total = time.time() - t_start

        print("Total time:", t_total, "s")
        if half_step:
            print("Collectors time:", t_collectors, "s")

        if half_step:
            return results
        else:
            return [collector(psi)]

    def __call__(self, psi, collector):

        # double step (to estimate the convergence)
        psi_double = self.thr.copy_array(psi)
        results_double = self._integrate(psi_double, False, collector)

        # actual integration
        results = self._integrate(psi, True, collector)

        # calculate the error (separately for each ensemble)
        batched_norm = lambda a: numpy.sqrt((numpy.abs(a) ** 2).sum(-1))
        psi_errors = batched_norm(psi_double.get() - psi.get()) / batched_norm(psi.get())
        print("Psi: mean err =", psi_errors.mean(), "max err =", psi_errors.max())

        # calculate result errors
        errors = dict(psi_strong_mean=psi_errors.mean(), psi_strong_max=psi_errors.max())
        for key in results[-1]:
            res_double = results_double[-1][key]
            res = results[-1][key]
            errors[key] = numpy.linalg.norm(res_double - res) / numpy.linalg.norm(res)
            print("Error in", key, "=", errors[key])

        return results, errors
