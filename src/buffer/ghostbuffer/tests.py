from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from . import gstbuf
from .gstbuf import GhostArray


class TestCreation(unittest.TestCase):
    def test_nothing(self):
        grr = GhostArray(0)
        self.assertEqual(0, grr.nbytes)

    def test_scalar(self):
        with self.assertRaises(ValueError) as cm:
            grr = GhostArray(1, creation="array")
        self.assertTrue(cm.exception.args[0], "zero dimension is not allowed")

    def test_zeros(self):
        grr = GhostArray(3, creation="zeros")
        self.assertEqual([0, 0, 0], list(grr.nda))

    def test_arange(self):
        grr = GhostArray(5, creation="arange")
        self.assertEqual(list(range(5)), list(grr.nda))

    def test_override(self):
        grr = GhostArray((3,4), gshape=1)
        # Make sure "nda" isn't writable.
        with self.assertRaises(AttributeError) as cm:
            grr.nda = np.arange(12).reshape((3,4))
        self.assertTrue(cm.exception.args[0].startswith(
            "attribute 'nda' of"))
        self.assertTrue(cm.exception.args[0].endswith(
            "objects is not writable"))
        # Make sure the addresses remains the same.
        grr.nda[...] = np.arange(12).reshape((3,4))
        self.assertEqual(grr._bodyaddr,
                         grr._ghostaddr + grr.itemsize * grr.offset)


class TestOffset(unittest.TestCase):
    def test_1d_vanilla(self):
        grr = GhostArray(10)
        self.assertEqual(0, grr.offset)

    def test_1d(self):
        grr = GhostArray(10, gshape=2)
        self.assertEqual(2, grr.offset)

    def test_2d(self):
        grr = GhostArray((10, 20), gshape=[2, 4])
        self.assertEqual(20*2 + 4, grr.offset)

    def test_3d(self):
        grr = GhostArray((10, 20, 30), gshape=[2, 4, 8])
        self.assertEqual(20*30*2 + 4*30 + 8, grr.offset)


class TestDrange(unittest.TestCase):
    def test_1d(self):
        grr = GhostArray(10, gshape=2)
        self.assertEqual((10,), grr.shape)
        self.assertEqual(((2,8),), grr.drange)
        for ntotal, (nghost, nbody) in zip(grr.shape, grr.drange):
            self.assertEqual(ntotal, nghost+nbody)

    def test_2d(self):
        grr = GhostArray((10,20), gshape=(2,4))
        self.assertEqual((10,20), grr.shape)
        self.assertEqual(((2,8),(4,16)), grr.drange)
        for ntotal, (nghost, nbody) in zip(grr.shape, grr.drange):
            self.assertEqual(ntotal, nghost+nbody)

    def test_2d(self):
        grr = GhostArray((10,20,30), gshape=(2,4,8))
        self.assertEqual((10,20,30), grr.shape)
        self.assertEqual(((2,8),(4,16),(8,22)), grr.drange)
        for ntotal, (nghost, nbody) in zip(grr.shape, grr.drange):
            self.assertEqual(ntotal, nghost+nbody)


class TestGshape1d(unittest.TestCase):
    def test_default(self):
        grr = GhostArray(10)
        self.assertEqual((0,), grr.gshape)
        self.assertEqual((10,), grr.bshape)
        self.assertEqual(((0,10),), grr.drange)

    def _check_negative(self, grr):
        self.assertEqual((0,), grr.gshape)
        self.assertEqual((10,), grr.bshape)
        self.assertEqual(((0,10),), grr.drange)

    def test_negative(self):
        self._check_negative(GhostArray(10, gshape=-1))

    def test_negative_iterable(self):
        self._check_negative(GhostArray(10, gshape=(-1,)))
        self._check_negative(GhostArray(10, gshape=[-1]))
        # Generator/iterator is OK.
        self._check_negative(GhostArray(10, gshape=(it for it in [-1])))

    def _check_larger(self, grr):
        self.assertEqual((10,), grr.gshape)
        self.assertEqual((0,), grr.bshape)
        self.assertEqual(((10,0),), grr.drange)

    def test_larger(self):
        self._check_larger(GhostArray(10, gshape=11))

    def test_larger_iterable(self):
        self._check_larger(GhostArray(10, gshape=(11,)))
        self._check_larger(GhostArray(10, gshape=[11]))
        # Generator/iterator is OK.
        self._check_larger(GhostArray(10, gshape=(it for it in [11])))

    def _check_longer(self, grr):
        self.assertEqual((3,), grr.gshape)
        self.assertEqual((7,), grr.bshape)
        self.assertEqual(((3,7),), grr.drange)

    def test_longer_iterable(self):
        self._check_longer(GhostArray(10, gshape=(3,5)))
        self._check_longer(GhostArray(10, gshape=[3,5]))
        # Generator/iterator is OK.
        self._check_longer(GhostArray(10, gshape=(it for it in [3,5])))


class TestGshape2d(unittest.TestCase):
    def test_default(self):
        grr = GhostArray((10,20))
        self.assertEqual((0,0), grr.gshape)
        self.assertEqual((10,20), grr.bshape)
        self.assertEqual(((0,10),(0,20)), grr.drange)

    def _check_negative(self, grr):
        self.assertEqual((0,0), grr.gshape)
        self.assertEqual((10,20), grr.bshape)
        self.assertEqual(((0,10),(0,20)), grr.drange)

    def test_negative(self):
        self._check_negative(GhostArray((10,20), gshape=-1))

    def test_negative_iterable(self):
        self._check_negative(GhostArray((10,20), gshape=(-1,)))
        self._check_negative(GhostArray((10,20), gshape=[-1]))
        # Generator/iterator is OK.
        self._check_negative(GhostArray((10,20), gshape=(it for it in [-1])))

    def _check_larger(self, grr):
        # This checks for shorter as well.
        self.assertEqual((10,0), grr.gshape)
        self.assertEqual((0,20), grr.bshape)
        self.assertEqual(((10,0),(0,20)), grr.drange)

    def test_larger(self):
        self._check_larger(GhostArray((10,20), gshape=11))

    def test_larger_iterable(self):
        self._check_larger(GhostArray((10,20), gshape=(11,)))
        self._check_larger(GhostArray((10,20), gshape=[11]))
        # Generator/iterator is OK.
        self._check_larger(GhostArray((10,20), gshape=(it for it in [11])))

    def _check_longer(self, grr):
        self.assertEqual((3,5), grr.gshape)
        self.assertEqual((7,15), grr.bshape)
        self.assertEqual(((3,7),(5,15)), grr.drange)

    def test_longer_iterable(self):
        self._check_longer(GhostArray((10,20), gshape=(3,5,7)))
        self._check_longer(GhostArray((10,20), gshape=[3,5,7]))
        # Generator/iterator is OK.
        self._check_longer(GhostArray((10,20), gshape=(it for it in [3,5,7])))


class TestParts(unittest.TestCase):
    def test_malformed(self):
        grr = GhostArray((10,20), gshape=(3,5))
        with self.assertRaises(ValueError) as cm:
            grr.ghostpart
        self.assertEqual(cm.exception.args[0], "malformed ghost shape")
        with self.assertRaises(ValueError) as cm:
            grr.bodypart
        self.assertEqual(cm.exception.args[0], "malformed ghost shape")

    def test_1d(self):
        grr = GhostArray(12, gshape=4, creation="arange")
        self.assertEqual(list(range(12)), list(grr.nda))
        self.assertEqual([3,2,1,0], list(grr.ghostpart))
        self.assertEqual(list(range(4,12)), list(grr.bodypart))

    def test_2d(self):
        grr = GhostArray((3,4), gshape=1)
        grr.nda[...] = np.arange(12).reshape((3,4))
        # Make sure the above writing doesn't override the memory holder.
        self.assertEqual(grr._bodyaddr,
                         grr._ghostaddr + grr.itemsize * grr.offset)
        # Check for value.
        self.assertEqual((1,4), grr.ghostpart.shape)
        self.assertEqual(list(range(4)), list(grr.ghostpart.ravel()))
        self.assertEqual((2,4), grr.bodypart.shape)
        self.assertEqual(list(range(4,12)), list(grr.bodypart.ravel()))


class TestRangedFill(unittest.TestCase):
    def test_1d(self):
        grr = GhostArray(5, gshape=2, dtype="int32")
        grr.ranged_fill()
        self.assertEqual((-1, -2), tuple(grr.ghostpart))
        self.assertEqual(( 0,  1, 2), tuple(grr.bodypart))
        self.assertEqual((-2, -1, 0, 1, 2), tuple(grr.nda))

    def test_2d(self):
        grr = GhostArray((5,3), gshape=2, dtype="int32")
        grr.ranged_fill()
        # Ghost part.
        self.assertEqual((2,3), grr.ghostpart.shape)
        self.assertEqual([-3,-2,-1], list(grr.ghostpart[0]))
        self.assertEqual([-6,-5,-4], list(grr.ghostpart[1]))
        # Body part.
        self.assertEqual((3,3), grr.bodypart.shape)
        self.assertEqual([0,1,2], list(grr.bodypart[0]))
        self.assertEqual([3,4,5], list(grr.bodypart[1]))
        self.assertEqual([6,7,8], list(grr.bodypart[2]))
        # Everything.
        self.assertEqual((5,3), grr.nda.shape)
        self.assertEqual([-6,-5,-4], list(grr.nda[0]))
        self.assertEqual([-3,-2,-1], list(grr.nda[1]))
        self.assertEqual([ 0, 1, 2], list(grr.nda[2]))
        self.assertEqual([ 3, 4, 5], list(grr.nda[3]))
        self.assertEqual([ 6, 7, 8], list(grr.nda[4]))

# vim: set fenc=utf8 ff=unix nobomb ai et sw=4 ts=4 tw=79:
