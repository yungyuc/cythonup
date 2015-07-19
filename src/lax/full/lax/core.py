from __future__ import absolute_import, division, print_function

import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from .march_cython import march_cython
from .march_c import march_c


"""
Example code that uses the Lax-Wendroff scheme to solve the simple advection
equation $u_t + a u_x = 0$.
"""


###############################################################################
# Here are the core of our numeric method.
###############################################################################
def march_python(cfl, sol, soln):
    """Pure Python implementation of Lax-Wendroff scheme"""
    itmax = sol.shape[0]-1
    it = 1
    while it < itmax:
        soln[it] = (1 - cfl*cfl) * sol[it]
        soln[it] += cfl * (cfl+1) / 2 * sol[it-1]
        soln[it] += cfl * (cfl-1) / 2 * sol[it+1]
        it += 1
    sol[:] = soln[:]


def march_numpy(cfl, sol, soln):
    """NumPy array-based implementation of Lax-Wendroff scheme"""
    soln[:] = sol[:]
    soln *= 1 - cfl*cfl
    soln[1:] += cfl * (cfl+1) / 2 * sol[:-1]
    soln[:-1] += cfl * (cfl-1) / 2 * sol[1:]
    sol[:] = soln[:]


###############################################################################
# Here goes the problem setup code.
###############################################################################
class HarmonicMotion(object):
    """Abstraction of moving the harmonic forward"""

    def __init__(self, number):
        """
        :param numer: Number of harmonic waves
        :ivar marcher: Must set this attribute to a function that really moves
            the wave.
        """
        # Construct grid.
        nx = 150
        xgrid = np.zeros(nx, dtype='float64')
        freq = 1
        xmin, xmax = -0.5*np.pi / freq, 4*np.pi / freq
        xgrid[:] = np.linspace(xmin, xmax, xgrid.shape[0])
        # Construct initial condition.
        sol = np.zeros_like(xgrid)
        slct = (xgrid > 0) & (xgrid < np.pi / freq)
        sol[slct] = np.sin(number * freq * xgrid)[slct]
        # Assign object attributes.
        self.xgrid = xgrid
        self.sol = sol
        self.soln = self.sol.copy()
        self.cfl = 1.0 # This cannot be greater than unity.
        self.time_increment = self.cfl * (xgrid[1] - xgrid[0]) # Stores as a record.
        self.marcher = None # This should be assigned from outside.

    @property
    def nx(self):
        """Number of points in the spatial grid."""
        return self.xgrid.shape[0]

    def march(self):
        """Call the marcher that moves the wave."""
        self.marcher(self.cfl, self.sol, self.soln)


###############################################################################
# Here are the runners.
###############################################################################
def run(marcher, steps=300, number=1):
    """Run simulation without output"""
    hm = HarmonicMotion(number)
    hm.marcher = marcher
    it = 0
    while (it < steps):
        hm.march()
        it += 1
    return hm


def demonstrate(marcher, args):
    """Demonstrate simulation with graphical output"""
    # Set up harmonic driver.
    hm = HarmonicMotion(args.number)
    hm.marcher = marcher

    # Set up plotter.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Location ($\\pi$)")
    ax.set_ylabel("Amplitude")
    # Title.
    titlefmt = "Step #%d/%d"
    text = ax.text(0.5, 1.05, "",
                   transform=ax.transAxes, va="center", ha="center")
    # Content line.
    line, = ax.plot(hm.xgrid/np.pi, hm.soln)

    if args.animate > 0:
        # Set up animator.
        def init():
            text.set_text(titlefmt % (0, args.steps))
        def update(num):
            """Update animation"""
            hm.march()
            text.set_text(titlefmt % (num+1, args.steps))
            line.set_ydata(hm.soln)
        ani = animation.FuncAnimation(
            fig, update, frames=args.steps, interval=args.animate,
            repeat=False, blit=False)
    else:
        # Loop to the end and draw.
        it = 0
        while it < args.steps:
            hm.march()
            it += 1
        text.set_text(titlefmt % (it, args.steps))
        line.set_ydata(hm.soln)

    # Show it.
    plt.show()


###############################################################################
# The last part, the command-line interface.
###############################################################################
def parse_command_line():
    """Parse the command-line arguments"""
    parser = argparse.ArgumentParser(description="Harmonic Motion")
    def wavenumber(val):
        val = float(val)
        assert val >= 1.0
        return val
    parser.add_argument("-s", dest="steps", action="store",
                        type=int, default=0,
                        help="Steps to run")
    parser.add_argument("-n", dest="number", action="store",
                        type=wavenumber, default=2,
                        help="Number of harmonics")
    parser.add_argument("-a", dest="animate", action="store",
                        type=float, default=0,
                        help="Animation interval (millisec)")
    parser.add_argument("-m", dest="marcher", action="store",
                        type=str, default="numpy",
                        help=("Select marcher from: "
                              "python, numpy, cython, and c; "
                              "default is %(default)s"))
    parser.add_argument("-c", dest="compute_only", action="store_true",
                        default=False,
                        help="Compute only; no plotting")
    return parser.parse_args()


def main():
    args = parse_command_line()
    marcher = globals()["march_"+args.marcher]

    if args.compute_only:
        run(marcher, steps=args.steps, number=args.number)
    else:
        demonstrate(marcher, args)


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
