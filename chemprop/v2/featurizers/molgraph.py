from dataclasses import InitVar, dataclass
from typing import NamedTuple, Protocol

import numpy as np
from rdkit import Chem

from chemprop.v2.featurizers.mixins import MolGraphFeaturizerMixin


class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""

    V: np.ndarray
    """an array of shape ``V x d_v`` containing the atom features of the molecule"""
    E: np.ndarray
    """an array of shape ``2 * E x d_e`` containing the bond features of the molecule"""
    edge_index: np.ndarray
    """an array of shape ``2 x E`` containing the edges of the graph in COO format"""
    rev_edge_index: np.ndarray
    """A vector of length ``E`` that maps from an edge index to the index of the source of the reverse edge in the ``edge_index`` attribute."""


class MoleculeMolGraphFeaturizerProto(Protocol):
    """A :class:`MoleculeMolGraphFeaturizerProto` featurizes RDKit molecules into
    :class:`MolGraph`s"""

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        """Featurize the input molecule into a molecular graph

        Parameters
        ----------
        mol : Chem.Mol
            the input molecule
        atom_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated atom features
        bond_features_extra : np.ndarray | None, default=None
            Additional features to concatenate to the calculated bond features

        Returns
        -------
        MolGraph
            the molecular graph of the molecule
        """


@dataclass
class MolGraphFeaturizer(MolGraphFeaturizerMixin, MoleculeMolGraphFeaturizerProto):
    """A :class:`MoleculeMolGraphFeaturizer` is the default implementation of a
    :class:`MoleculeMolGraphFeaturizerProto`

    Parameters
    ----------
    atom_featurizer : AtomFeaturizerProto, default=AtomFeaturizer()
        the featurizer with which to calculate feature representations of the atoms in a given
        molecule
    bond_featurizer : BondFeaturizerProto, default=BondFeaturizer()
        the featurizer with which to calculate feature representations of the bonds in a given
        molecule
    bond_messages : bool, default=True
        whether to prepare the `MolGraph`s for use with message passing on bonds
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated
        features of each bond
    """

    extra_atom_fdim: InitVar[int] = 0
    extra_bond_fdim: InitVar[int] = 0

    def __post_init__(self, extra_atom_fdim: int = 0, extra_bond_fdim: int = 0):
        super().__post_init__()

        self.atom_fdim += extra_atom_fdim
        self.bond_fdim += extra_bond_fdim

    def __call__(
        self,
        mol: Chem.Mol,
        atom_features_extra: np.ndarray | None = None,
        bond_features_extra: np.ndarray | None = None,
    ) -> MolGraph:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )

        X_v = np.array([self.atom_featurizer(a) for a in mol.GetAtoms()])
        X_e = np.empty((2 * n_bonds, self.bond_fdim))
        edge_index = [[], []]

        if atom_features_extra is not None:
            X_v = np.hstack((X_v, atom_features_extra))

        i = 0
        for u in range(n_atoms):
            for v in range(u + 1, n_atoms):
                bond = mol.GetBondBetweenAtoms(u, v)
                if bond is None:
                    continue

                x_e = self.bond_featurizer(bond)
                if bond_features_extra is not None:
                    x_e = np.concatenate((x_e, bond_features_extra[bond.GetIdx()]))

                X_e[i : i + 2] = x_e

                edge_index[0].extend([u, v])
                edge_index[1].extend([v, u])

                i += 2

        rev_edge_index = np.arange(len(X_e)).reshape(-1, 2)[:, ::-1].ravel()
        edge_index = np.array(edge_index, int)

        return MolGraph(X_v, X_e, edge_index, rev_edge_index)
