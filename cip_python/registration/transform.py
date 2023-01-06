"""
File: transform.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    This module contains usefull classes to work with ITK transforms, for
    example the Composite transform.
"""
import SimpleITK as sitk


class CompositeTransform():
    """
    Composite transforms represents a stack of transformtation. The
    transforms are composed in reverse order with the back being applied first.
    For example, (T0 o T1) (x) = T0(T1(x)) an the transforms are stored in a
    queue as [T0, T1, ....]. the method AddTransform() adds the transforms to
    the back of the queue. For example let be A the affine part of a complex
    free form deformation, T, and F the pure free form deformation without an
    affine component, i.e F x = A^-1 T x. Now T x = A F x, so to compose a
    complet tx T, from a pure ffd and an affine tranasformation we need to do:
    T = A o F

    comp = CompisteTransform()
    comp.AddTransform(affine)      ----> [affine]
    comp.AddTransform(ffd)   ----> [affine, ffd]

    comp.TransformPoint(x)      ----> affine(ffd(x))

    Note: The first tx added is the last to be applied.
    """

    def __new__(self, tx):
        dim = tx[0].GetDimension()
        composite = sitk.CompositeTransform(dim)
        for tx_i in tx:
            composite.AddTransform(tx_i)
        return composite
