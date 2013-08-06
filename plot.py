import os
import numpy
import mplhelpers as mpl
import matplotlib.pyplot as plt
import scipy.ndimage
from mpl_toolkits.axes_grid1 import ImageGrid

from wigner import get


r_bohr = 5.2917720859e-11 # Bohr radius
hbar = 1.054571628e-34 # Planck constant
a_11 = 100. # scattering length
m_rb = 1.443160648e-25 # mass of the Rb atom, kg


cmap = mpl.cm_heightmap
suffix = "_grayscale.pdf" if mpl.grayscale else ".pdf"


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def plot_1d_no_coupling_varying_L():

    gamma = 0.1
    L_base = 10.
    modes_base = 32
    t = 25.

    fig = mpl.figure()

    colors = [mpl.color.f[c].main for c in ('blue', 'red', 'green', 'yellow')]
    linestyles = ['-', '--', '-.', ':']
    print "* 1d_no_coupling_varying_L"

    s = fig.add_subplot(111, xlabel='$\\mathit{\\tau}$', ylabel='$\\mathit{J}$')

    s.plot([0, t], [0, 0], color='grey', linestyle='--', linewidth=0.5)

    for i, j in enumerate([0, 1, 2, 3]):
        L = L_base * 2 ** j
        modes = modes_base * 2 ** j
        results = get('1d',
            initial='opposite',
            gamma=gamma, L_trap=L, t=t, nu=0,
            steps=10000, modes=modes, ensembles=1000)

        N = L / gamma
        times = numpy.linspace(0, results['t'], results['samples'] + 1)
        n0m = numpy.array(results['Nminus_mean']) / N
        n0e = numpy.array(results['Nminus_std']) / numpy.sqrt(results['ensembles']) / N
        n1m = numpy.array(results['Nplus_mean']) / N
        n1e = numpy.array(results['Nplus_std']) / numpy.sqrt(results['ensembles']) / N
        print L, results['errors']['Nminus_mean']

        ax = s.plot(times, (n1m - n0m) / 4, label="$L={L},\\,M={M}$".format(L=L, M=modes),
            color=colors[i], linestyle=linestyles[i], dashes=mpl.dash[linestyles[i]])
        #s.plot(times, n0m + n0e, color=ax[0].get_color(), linestyle='--')
        #s.plot(times, n0m - n0e, color=ax[0].get_color(), linestyle='--')

    s.set_ylim(-0.5, 0.5)
    s.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    #s.legend()
    fig.tight_layout()
    fig.savefig('1d_no_coupling_varying_L' + suffix)


def plot_1d_varying_coupling():

    gamma = 0.1
    L = 80.
    modes = 256
    t = 25.

    colors = [mpl.color.f[c].main for c in ('blue', 'red', 'green', 'yellow')]
    linestyles = ['-', '--', '-.', ':']
    print "* 1d_varying_coupling"

    fig = mpl.figure()
    s = fig.add_subplot(111, xlabel='$\\mathit{\\tau}$', ylabel='$\\mathit{J}$')

    s.plot([0, t], [0, 0], color='grey', linestyle='--', linewidth=0.5)

    for i, nu in enumerate([0.0, 0.01, 0.1, 1]):
        results = get('1d',
            initial='opposite',
            gamma=gamma, L_trap=L, t=t, nu=nu,
            steps=10000, modes=modes, ensembles=1000)

        N = L / gamma
        times = numpy.linspace(0, results['t'], results['samples'] + 1)
        n0m = numpy.array(results['Nminus_mean']) / N
        n0e = numpy.array(results['Nminus_std']) / numpy.sqrt(results['ensembles']) / N
        n1m = numpy.array(results['Nplus_mean']) / N
        n1e = numpy.array(results['Nplus_std']) / numpy.sqrt(results['ensembles']) / N
        print nu, results['errors']['Nminus_mean']

        #ax = s.plot(times, n0m, label="$\\nu = {nu}$".format(nu=nu),
        #    color=colors[i], linestyle=linestyles[i], dashes=mpl.dash[linestyles[i]])
        ax = s.plot(times, (n1m - n0m) / 4, label="$\\nu = {nu}$".format(nu=nu),
            color=colors[i], linestyle=linestyles[i], dashes=mpl.dash[linestyles[i]])
        #s.plot(times, n0m + n0e, ax[0].get_color() + '--')
        #s.plot(times, n0m - n0e, ax[0].get_color() + '--')

    s.set_ylim(-0.5, 0.5)
    s.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    #s.legend()
    fig.tight_layout()
    fig.savefig('1d_varying_coupling' + suffix)


def plot_1d_varying_g12():

    gamma = 0.1
    L = 80.
    modes = 256
    t = 25.

    colors = [mpl.color.f[c].main for c in ('blue', 'red', 'green', 'yellow')]
    linestyles = ['-', '--', '-.', ':']
    print "* 1d_varying_g12"

    fig = mpl.figure()
    s = fig.add_subplot(111, xlabel='$\\mathit{\\tau}$', ylabel='$\\mathit{J}$')

    s.plot([0, t], [0, 0], color='grey', linestyle='--', linewidth=0.5)

    for i, U12 in enumerate([0, 0.3, 0.5, 0.7]):
        results = get('1d',
            initial='opposite',
            gamma=gamma, L_trap=L, t=t, nu=0.1,
            steps=10000, modes=modes, ensembles=1000, U12=U12)

        N = L / gamma
        times = numpy.linspace(0, results['t'], results['samples'] + 1)
        n0m = numpy.array(results['Nminus_mean']) / N
        n0e = numpy.array(results['Nminus_std']) / numpy.sqrt(results['ensembles']) / N
        n1m = numpy.array(results['Nplus_mean']) / N
        n1e = numpy.array(results['Nplus_std']) / numpy.sqrt(results['ensembles']) / N
        print U12, results['errors']['Nminus_mean']

        #ax = s.plot(times, n0m, label="$\\nu = {nu}$".format(nu=nu),
        #    color=colors[i], linestyle=linestyles[i], dashes=mpl.dash[linestyles[i]])
        ax = s.plot(times, (n1m - n0m) / 4, label="$\\g_{{12}} = {U12} g_{{11}}$".format(U12=U12),
            color=colors[i], linestyle=linestyles[i], dashes=mpl.dash[linestyles[i]])
        #s.plot(times, n0m + n0e, ax[0].get_color() + '--')
        #s.plot(times, n0m - n0e, ax[0].get_color() + '--')

    s.set_ylim(-0.5, 0.5)
    s.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    #s.legend()
    fig.tight_layout()
    fig.savefig('1d_varying_g12' + suffix)


def plot_1d_density():

    gamma = 0.1
    L = 80.
    modes = 256
    N = L / gamma

    results = get('1d',
        initial='opposite',
        gamma=gamma, L_trap=L, t=75., nu=0.1,
        steps=50000, modes=modes, ensembles=1, samples=250)

    n = numpy.array(results['density']).transpose(1, 0, 2)
    print "* 1d_density"
    print results['errors']['density']
    times = numpy.linspace(0, results['t'], results['samples'] + 1)

    fig = mpl.figure()
    s = fig.add_subplot(111, xlabel='$\\mathit{\\tau}$', ylabel='$\\mathit{z}$')

    data = (n[0] - n[1]).T / (4 * N / L)
    data = numpy.clip(data, -0.5, 0.5)

    levels = numpy.linspace(-0.5, 0.5, 11)
    #data = scipy.ndimage.zoom(data, 0.5)
    data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)

    im = s.imshow(data,
        extent=(0, times[-1], -L/2, L/2),
        vmin=-0.5, vmax=0.5,
        cmap=cmap,
        interpolation='none',
        aspect=(results['t']) / L / 1.6)

    fig.colorbar(im,orientation='vertical', shrink=0.8, ticks=levels).set_label('$\\mathit{j}$')
    fig.tight_layout()
    fig.savefig('1d_density' + suffix)


def plot_2d_varying_coupling():

    gamma = 0.1
    L = 80.
    modes = 256
    t = 25.

    colors = [mpl.color.f[c].main for c in ('blue', 'red', 'green', 'yellow')]
    linestyles = ['-', '--', '-.', ':']
    print "* 2d_varying_coupling"

    fig = mpl.figure()
    s = fig.add_subplot(111, xlabel='$\\mathit{\\tau}$', ylabel='$\\mathit{J}$')
    s.plot([0, t], [0, 0], color='grey', linestyle='--', linewidth=0.5)

    for i, nu in enumerate([0.001, 0.01, 0.1, 1]):
        results = get('2d',
            initial='opposite',
            gamma=gamma, L_trap=L, t=t, nu=nu,
            steps=4000, modes=modes, ensembles=200)

        N = L ** 2 / gamma
        times = numpy.linspace(0, results['t'], results['samples'] + 1)
        n0m = numpy.array(results['Nminus_mean']) / N
        n0e = numpy.array(results['Nminus_std']) / numpy.sqrt(results['ensembles']) / N
        n1m = numpy.array(results['Nplus_mean']) / N
        n1e = numpy.array(results['Nplus_std']) / numpy.sqrt(results['ensembles']) / N
        print nu, results['errors']['Nminus_mean']

        ax = s.plot(times, (n1m - n0m) / 4, label="$\\nu = {nu}$".format(nu=nu),
            color=colors[i], linestyle=linestyles[i], dashes=mpl.dash[linestyles[i]])
        #s.plot(times, ((n1m - n0m) / 4 + n0e + n1e), color=ax[0].get_color(), linestyle='--')
        #s.plot(times, ((n1m - n0m) / 4 - n0e - n1e), color=ax[0].get_color(), linestyle='--')

    s.set_ylim(-0.5, 0.5)
    s.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    fig.tight_layout()
    fig.savefig('2d_varying_coupling' + suffix)


def plot_2d_density():

    modes = 128
    L = 80.
    gamma = 0.1
    N = L ** 2 / gamma
    samples = 4

    results = get('2d',
        initial='opposite',
        gamma=gamma, L_trap=L, t=20., nu=0.1,
        steps=5000, modes=modes, ensembles=1, samples=samples)

    n = numpy.array(results['density']).transpose(1, 0, 2, 3)
    print "* 2d_density"
    print results['errors']['density']
    times = numpy.linspace(0, results['t'], results['samples'] + 1)
    levels = numpy.linspace(-0.5, 0.5, 11)

    fig = mpl.figure(width=2, aspect=0.35)

    grid2 = ImageGrid(fig, [0.1, 0.0, 0.8, 1.0],
                          nrows_ncols = (1, 3),
                          direction="row",
                          axes_pad = 0.2,
                          add_all=True,
                          label_mode = "1",
                          share_all = True,
                          cbar_location="right",
                          cbar_mode="single",
                          cbar_size="5%",
                          cbar_pad=0.2,
                          )
    grid2[0].set_xlabel('$\\mathit{y}$')
    grid2[0].set_ylabel('$\\mathit{x}$')

    for j, nt in enumerate([1, 2, 4]):
        #s = fig.add_subplot([311, 321, 331][j], xlabel='$\\mathit{y}$', ylabel='$\\mathit{x}$')
        s = grid2[j]

        data = (n[0][nt] - n[1][nt]).T / (4 * N / (L**2))
        data = numpy.clip(data, -0.5, 0.5)
        #data = scipy.ndimage.zoom(data, 0.25)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)

        im = s.imshow(data,
            extent=(-L/2, L/2, -L/2, L/2),
            vmin=-0.5, vmax=0.5,
            cmap=cmap,
            interpolation='none',
            aspect=1)

        fig.text(0.18 + 0.26 * j, 0.9, '$\\tau=' + str(int(times[nt])) + '$')

    #fig.colorbar(im,orientation='vertical', shrink=0.6, ticks=levels).set_label('$\\mathit{j}$')
    #fig.tight_layout()
    grid2[-1].cax.colorbar(im, ticks=levels).set_label_text('$\\mathit{j}$')
    grid2[-1].cax.toggle_label(True)
    fig.savefig('2d_density' + suffix)


def plot_contents_pic():

    modes = 128
    L = 80.
    gamma = 0.1
    N = L ** 2 / gamma
    samples = 4

    results = get('2d',
        initial='opposite',
        gamma=gamma, L_trap=L, t=20., nu=0.1,
        steps=5000, modes=modes, ensembles=1, samples=samples)

    n = numpy.array(results['density']).transpose(1, 0, 2, 3)
    print "* contents (2d density)"
    print results['errors']['density']
    times = numpy.linspace(0, results['t'], results['samples'] + 1)
    levels = numpy.linspace(-0.5, 0.5, 11)

    nt = 2
    fig = mpl.figure(width=2./3, aspect=0.9)
    s = fig.add_subplot(111)
    s.set_xticks([])
    s.set_yticks([])

    data = (n[0][nt] - n[1][nt]).T / (4 * N / (L**2))
    data = numpy.clip(data, -0.5, 0.5)
    #data = scipy.ndimage.zoom(data, 0.25)
    data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)

    im = s.imshow(data,
        extent=(-L/2, L/2, -L/2, L/2),
        vmin=-0.5, vmax=0.5,
        cmap=cmap,
        interpolation='none',
        aspect=1)

    fig.tight_layout()
    fig.savefig('contents' + suffix)


def plot_3d_density():

    modes = 128
    L = 80.
    gamma = 0.1
    N = L ** 3 / gamma
    samples = 4

    results = get('3d',
        initial='opposite',
        gamma=gamma, L_trap=L, t=20., nu=0.1,
        steps=5000, modes=modes, ensembles=1, samples=samples)

    n = numpy.array(results['slice']).transpose(1, 0, 2, 3)
    print "* 3d_density"
    print results['errors']['slice']
    times = numpy.linspace(0, results['t'], results['samples'] + 1)
    levels = numpy.linspace(-0.5, 0.5, 11)

    fig = mpl.figure(width=2, aspect=0.35)

    grid2 = ImageGrid(fig, [0.1, 0.0, 0.8, 1.0],
                          nrows_ncols = (1, 3),
                          direction="row",
                          axes_pad = 0.2,
                          add_all=True,
                          label_mode = "1",
                          share_all = True,
                          cbar_location="right",
                          cbar_mode="single",
                          cbar_size="5%",
                          cbar_pad=0.2,
                          )
    grid2[0].set_xlabel('$\\mathit{y}$')
    grid2[0].set_ylabel('$\\mathit{x}$')

    for j, nt in enumerate([1, 2, 4]):
        #s = fig.add_subplot([311, 321, 331][j], xlabel='$\\mathit{y}$', ylabel='$\\mathit{x}$')
        s = grid2[j]

        data = (n[0][nt] - n[1][nt]).T / (4 * N / (L**3))
        data = numpy.clip(data, -0.5, 0.5)
        #data = scipy.ndimage.zoom(data, 0.25)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)

        im = s.imshow(data,
            extent=(-L/2, L/2, -L/2, L/2),
            vmin=-0.5, vmax=0.5,
            cmap=cmap,
            interpolation='none',
            aspect=1)

        label = ['$\\tau=5$', '$\\tau=10$', '$\\tau=20$']
        fig.text(0.18 + 0.26 * j, 0.9, label[j])

    #fig.colorbar(im,orientation='vertical', shrink=0.6, ticks=levels).set_label('$\\mathit{j}$')
    #fig.tight_layout()
    grid2[-1].cax.colorbar(im, ticks=levels).set_label_text('$\\mathit{j}$')
    grid2[-1].cax.toggle_label(True)
    fig.savefig('3d_density' + suffix)

    """
    comp = 0
    for nt in range(samples+1):
        fig = mpl.figure(aspect=0.9)
        s = fig.add_subplot(111, xlabel='$\\mathit{y}$', ylabel='$\\mathit{z}$')

        data = (n[0][nt] - n[1][nt]).T / (4 * N / (L ** 3))
        data = numpy.clip(data, -0.5, 0.5)
        levels = numpy.linspace(-0.5, 0.5, 11)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)

        im = s.contourf(
            data,
            extent=(-L/2, L/2, -L/2, L/2),
            cmap=cmap,
            extend='both',
            antialiased=False,
            aspect=1,
            levels=levels)

        fig.colorbar(im,orientation='vertical', shrink=0.6, ticks=levels).set_label('$\\mathit{j}$')
        fig.tight_layout()
        fig.savefig('3d_density0_t' + ('%2.1f' % times[nt]) + suffix)
    """

def plot_2d_density_movie():

    modes = 128
    L = 80.
    gamma = 0.01
    nu = 0.01
    t = 300.

    N = L ** 2 / gamma
    samples = 500

    results = get('2d',
        initial='opposite',
        gamma=gamma, L_trap=L, t=t, nu=nu,
        steps=40000, modes=modes, ensembles=1, samples=samples)

    n = numpy.array(results['density']).transpose(1, 0, 2, 3)
    n_nobs = numpy.array(results['density_nobs']).transpose(1, 0, 2, 3)

    print "* 2d_density"
    print results['errors']['density']
    times = numpy.linspace(0, results['t'], results['samples'] + 1)
    levels = numpy.linspace(-0.5, 0.5, 11)

    comp = 0
    for nt in range(samples+1):
        """
        fig = mpl.figure(aspect=0.9)
        s = fig.add_subplot(111, xlabel='$\\mathit{y}$', ylabel='$\\mathit{x}$')

        data = (n[0][nt] - n[1][nt]).T / (4 * N / (L**2))
        data = numpy.clip(data, -0.5, 0.5)

        levels = numpy.linspace(-0.5, 0.5, 11)
        #data = scipy.ndimage.zoom(data, 0.5)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)

        im = s.contourf(
            data,
            extent=(-L/2, L/2, -L/2, L/2),
            cmap=cmap,
            extend='both',
            antialiased=False,
            aspect=1,
            levels=levels)

        t = add_inner_title(s, "$\\tau = %d$" % int(times[nt]), loc=3)
        t.patch.set_alpha(0.5)

        fig.tight_layout()
        fig.savefig('temp/2d_density0_%03d.png' % nt)
        """

        kwds = dict(
            extent=(-L/2, L/2, -L/2, L/2),
            cmap=cmap,
            interpolation='none',
            aspect=1)

        fig = mpl.figure(width=2, aspect=0.35)

        grid2 = ImageGrid(fig, [0.1, 0.0, 0.8, 1.0],
                              nrows_ncols = (1, 3),
                              direction="row",
                              axes_pad = 0.25,
                              add_all=True,
                              label_mode = "1",
                              share_all = True,
                              cbar_location="top",
                              cbar_mode="each",
                              cbar_size="7%",
                              cbar_pad="1%",
                              )
        grid2[0].set_xlabel('$\\mathit{y}$')
        grid2[0].set_ylabel('$\\mathit{x}$')

        s = grid2[0]
        data = (n[0][nt] - n[1][nt]).T / (4 * N / (L**2))
        data = numpy.clip(data, -0.5, 0.5)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)
        im = s.imshow(data, vmin=-0.5, vmax=0.5, **kwds)
        t = add_inner_title(s, "$j$, $\\tau = %d$" % int(times[nt]), loc=3)
        t.patch.set_alpha(0.5)
        grid2[0].cax.colorbar(im)

        s = grid2[1]
        data = (n_nobs[0][nt]).T / (N / (L**2))
        data = numpy.clip(data, 0, 1.5)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)
        im = s.imshow(data, vmin=0, vmax=1.5, **kwds)
        t = add_inner_title(s, "$|\\psi_0|^2$", loc=3)
        t.patch.set_alpha(0.5)
        grid2[1].cax.colorbar(im)

        s = grid2[2]
        data = (n_nobs[1][nt]).T / (N / (L**2))
        data = numpy.clip(data, 0, 1.5)
        data = scipy.ndimage.gaussian_filter(data, sigma=0.5, order=0)
        im = s.imshow(data, vmin=0, vmax=1.5, **kwds)
        t = add_inner_title(s, "$|\\psi_1|^2$", loc=3)
        t.patch.set_alpha(0.5)
        grid2[2].cax.colorbar(im)

        fig.savefig('temp/2d_density0_%03d.png' % nt, dpi=200)


    os.system("./mencoder.sh 'mf://temp/2d_density0_*.png' 2d_density.avi")

if __name__ == '__main__':
    plot_1d_no_coupling_varying_L()
    plot_1d_varying_coupling()
    plot_1d_varying_g12()
    plot_2d_varying_coupling()
    plot_1d_density()
    plot_2d_density()
    plot_3d_density()
    plot_contents_pic()

    #plot_2d_density_movie()
