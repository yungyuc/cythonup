from __future__ import absolute_import, division, print_function

import argparse

import numpy as np
from matplotlib import pyplot as plt

from .march_c import march_c


"""
Example code that uses the Lax-Wendroff scheme to solve the simple advection
equation $u_t + a u_x = 0$.
"""


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
# Here is the runner.
###############################################################################
def show_step(marcher, number, steps, **kw):
    """Demonstrate simulation with graphical output"""
    # Set up harmonic driver.
    hm = HarmonicMotion(number)
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

    # Loop to the end and draw.
    it = 0
    while it < steps:
        hm.march()
        it += 1
    text.set_text(titlefmt % (it, steps))
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
                        help="Number of harmonics; default is %(default)s")
    return parser.parse_args()


def main():
    args = parse_command_line()
    kw = vars(args)
    show_step(march_c, **kw)


if __name__ == '__main__':
    main()

# vim: set ff=unix fenc=utf8 nobomb ai et sw=4 ts=4 tw=79:
