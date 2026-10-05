"""A real MMFF94 oracle: each call constructs and evaluates one geometry."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

GRID = np.arange(-180, 180, 10, dtype=float)
CANDIDATES = np.array([(a, b) for a in GRID for b in GRID])
TORSIONS = ((0, 1, 2, 3), (1, 2, 3, 4))
GEOMETRY_SEED = 2026


class PentaneEnergy:
    """Independent copies of one embedded geometry; no relaxation or energy cache.

    SMILES atom indices 0..4 correspond to carbons C1..C5. Hydrogens are
    explicit. ETKDG embedding occurs once, outside the search budget, and
    the initial geometry is never force-field optimized.
    """

    def __init__(self, geometry_seed=GEOMETRY_SEED):
        self.geometry_seed = geometry_seed
        self.template = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
        params = AllChem.ETKDGv3()
        params.randomSeed = geometry_seed
        params.numThreads = 1
        if AllChem.EmbedMolecule(self.template, params) != 0:
            raise RuntimeError("RDKit could not embed pentane")
        self.properties = AllChem.MMFFGetMoleculeProperties(
            self.template, mmffVariant="MMFF94"
        )
        if self.properties is None:
            raise RuntimeError("Pentane has missing MMFF94 parameters")
        self.evaluations = 0

    def geometry(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (2,) or not np.all(np.isfinite(angles)):
            raise ValueError("Expected two finite torsion angles in degrees")
        mol = Chem.Mol(self.template)
        conf = mol.GetConformer()
        for indices, angle in zip(TORSIONS, angles):
            rdMolTransforms.SetDihedralDeg(conf, *indices, float(angle))
        return mol

    def __call__(self, angles):
        mol = self.geometry(angles)
        # Recreate the force field after rotations so its positions are current.
        # Include all nonbonded pairs even in severe steric clashes.
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, self.properties, nonBondedThresh=100.0, confId=0
        )
        if ff is None:
            raise RuntimeError("Could not construct MMFF94 force field")
        energy = float(ff.CalcEnergy())
        self.evaluations += 1
        if not np.isfinite(energy):
            raise RuntimeError("MMFF94 returned a nonfinite energy")
        return energy
