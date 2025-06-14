import numpy as np
from scipy.linalg import null_space
from sympy import symbols, Matrix, eye, roots, sqrt
from functools import cached_property

class Huckel:

    def __init__(self, structure: dict[int, list] | str, charge: int = 0):
        """
        :param structure: dict[int, list]: {atom: [connected_atoms]} or str in SMILES format
        :param charge: int: net charge of the molecule, default = 0
        """
        if isinstance(structure, str):
            structure = self._smiles_to_structure(structure)
        self.structure = structure
        self.check_structure()
        self.n = len(structure.keys())
        self.e = self.n - charge
        self.atoms = list(structure.keys())
        self.trans = {self.atoms[i]: i for i in range(self.n)}
    
    @staticmethod
    def _smiles_to_structure(smiles: str) -> dict[int, list]:
        """
        Convert a SMILES string to a structure dictionary.
        This is a placeholder function, you need to implement the actual conversion logic.
        :param smiles: str: SMILES string
        :return: dict[int, list]: {atom: [connected_atoms]}
        """
        
        from rdkit import Chem
        from collections import defaultdict

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES string: {smiles}')
        
        structure = defaultdict(list)
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtomIdx() + 1
            atom2 = bond.GetEndAtomIdx() + 1
            structure[atom1].append(atom2)
            structure[atom2].append(atom1)
        
        return dict(structure)

    def check_structure(self):
        """Check the structure for validity."""
        found = set()
        bonds = []
        for atom in self.structure:
            if len(self.structure[atom]) > 3:
                raise ValueError(f'Atom {atom} in the structure has more than 3 bonds ({len(self.structure[atom])})')
            if len(self.structure[atom]) < 1:
                raise ValueError(f'Atom {atom} in the structure has no connected atoms')
            for catom in self.structure[atom]:
                s = frozenset([atom, catom])
                if s in found:
                    found.remove(s)
                    bonds.append(tuple(s))
                else:
                    found.add(s)
        if found:
            raise ValueError(f'Some Bonds are not mutual ({found})')

    @cached_property
    def hamiltonian(self) -> Matrix:
        """
        Construct the Hamiltonian matrix of the Huckel model. 
        Diagonal elements are x, off-diagonal elements are 1 if the atoms are connected, otherwise 0.
        :return: Hamiltonian matrix of the Huckel model
        """
        x = symbols('x')
        B = Matrix([[int(j in self.structure[i]) for j in self.atoms] for i in self.atoms])
        A: Matrix = x * eye(self.n) + B
        return A

   @cached_property
    def det_hamiltonian(self):
        A = self.hamiltonian
        p = A.charpoly()
        det_A = p.as_poly().subs(p.gens[0], 0)
        return det_A

    @cached_property
    def energy(self) -> dict:
        """
        Calculate the energy levels of the Huckel model.
        return values are the x, which represents the energy in the from of α-βx, where α is the Coulomb integral and β is the resonance integral.
        :return: dict[energy, multiplicity], sorted by increasing energy
        """
        det_A = self.det_hamiltonian
        try: 
            _r: dict = roots(det_A)
            assert sum(_r.values()) == self.n
            # The sum of the multiplicity is not equal to the number of atoms, otherwise use nroots. 
        except AssertionError:
            _r: list = nroots(det_A, maxsteps=500)
            _rr = []
            for z in _r:
                if abs(z.as_real_imag()[1]) < 1e-9:
                    _rr.append(z.as_real_imag()[0])
            _r = {x: _rr.count(x) for x in _rr}
            assert sum(_r) == self.n, "For certain reason the number of roots is not equal to the number of atoms"
        
        if self.n > 20:
            r = { k.evalf(chop=True): _r[k]  for k in _r }
        else:
            r = _r
        return {
            k: r[k] 
            for k in sorted(r.keys())
        }
    
    @cached_property
    def xi(self) -> list:
        """
        :return: list[energy]: sorted list of energy levels with multiplicity
        """
        r = self.energy
        ret = []
        for k, v in r.items():
            ret += [k] * v
        return sorted(ret)
    
    @cached_property
    def orbitals(self) -> list[list]:
        """
        :return: list[list[coefficients]]: list of orbitals, each orbital is a list of coefficients for each atom
        """
        ret = []
        for x, g in self.energy.items():
            if self.n < 20:
                # For small systems, use sympy's nullspace
                D: Matrix = self.hamiltonian.subs(symbols('x'), x)
                _null_space = D.nullspace()
                for i in range(g):
                    c = _null_space[i].normalized().tolist()
                    c = [c[j][0] for j in range(self.n)]
                    ret.append(c)
            else:
                # For larger systems, use numpy's null_space, digitalize the matrix
                D = self.hamiltonian.subs(symbols('x'), x).applyfunc(lambda x: x.evalf()).tolist()
                D = np.array(D, dtype=np.float64)
                ns = null_space(D)
                if ns.shape[1] > 0:
                    for i in range(g):
                        c = ns[:, i]
                        c = c / np.linalg.norm(c)
                        c = c.tolist()
                        ret.append(c)
        return ret
    
    @cached_property
    def electrons(self) -> list[int]:
        """
        :return: list[int]: list of electrons in each orbital, where each orbital can hold 2 electrons
        """
        ret = []
        e = self.e
        for x, g in self.energy.items():
            maxes = 2 * g
            if e >= maxes:
                ret += [2] * g
                e -= maxes
            else:
                if e > g:
                    ret += [2] * (e - g) + [1] * (2 * g - e)
                elif e == g:
                    ret += [1] * g
                elif e == 0:
                    ret += [0] * g
                else:
                    ret += [1] * e + [0] * (g - e)
                e = 0
        return ret

    @cached_property
    def HOMO(self):
        """
        :return: the energy of the highest occupied molecular orbital (HOMO) showing in x
        """
        H = None
        for i, e in enumerate(self.electrons):
            if e > 0:
                H = i
            else:
                return self.xi[H]
    
    @cached_property
    def LUMO(self):
        """
        :return: the energy of the lowest unoccupied molecular orbital (LUMO) showing in x
        """
        for i, e in enumerate(self.electrons[::-1]):
            if e == 0:
                return self.xi[i]

    @cached_property
    def gap_energy(self):
        """
        :return: the energy gap showing in x
        """
        return self.HOMO - self.LUMO
    
    @cached_property
    def electron_density(self) -> list:
        """
        :return: list[density]: list of electron density for each atom
        """
        return [
            sum(self.electrons[j] * (self.orbitals[j][i] ** 2) 
                for j in range(self.n)) 
            for i in range(self.n)
            ]
    
    @cached_property
    def atom_charge(self):
        """
        :return: list[charge]: list of charge for each atom
        """
        return [1 - self.electron_density[i] for i in range(self.n)]
    
    def bond_order(self, atom_1: int, atom_2: int):
        """
        :param atom_1: int: atom 1
        :param atom_2: int: atom 2
        :return: pi-bond order of the bond between atom 1 and atom 2
        """
        if atom_1 not in self.atoms:
            raise ValueError(f'Atom {atom_1} is not in the structure')
        if atom_2 not in self.atoms:
            raise ValueError(f'Atom {atom_2} is not in the structure')
        if atom_2 not in self.structure[atom_1]:
            # Not-connected atoms
            return 0
        return sum(
            self.electrons[j] * self.orbitals[j][atom_1 - 1] * self.orbitals[j][atom_2 - 1] 
            for j in range(self.n)
            )
    
    @cached_property
    def bond_orders(self):
        """
        :return: dict[(atom1, atom2), bond_order]: dictionary of bond orders for each pair of atoms
        """
        return {
            (atom1, atom2): self.bond_order(atom1, atom2)
            for atom1 in self.atoms 
            for atom2 in self.atoms 
            if atom2 in self.structure[atom1] 
                and atom1 < atom2
            }
    
    def free_valance(self, atom: int):
        """
        :param atom: int: atom to calculate the free valance for
        :return: float: free valance of the atom, with Fmax = sqrt(3)
        """
        return sqrt(3) - sum(
            self.bond_order(atom, atom2) 
            for atom2 in self.atoms 
            if atom2 != atom
            )
    
    @cached_property
    def free_valances(self):
        """
        :return: list[free_valance]: list of free valances for each atom
        """
        return [
            self.free_valance(atom) 
            for atom in self.atoms
            ]
    
    def plot_energy(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        color = 'black'
        line_length = 0.3
        elec_index = 0
        for i, (energy, g) in enumerate(self.energy.items()):
            spacing = 0.5
            for j in range(g):
                x_start = -line_length/2 + (j-(g-1)/2)*spacing
                x_end = x_start + line_length
                ax.hlines(energy.evalf(chop=True), x_start, x_end, color=color, linewidth=line_length)
                elec_pos = x_start + line_length/2
                elec_num = self.electrons[elec_index]
                if elec_num == 1:
                    ax.text(elec_pos-0.06, energy, '↑', fontsize=12, verticalalignment='center')
                elif elec_num == 2:
                    ax.text(elec_pos-0.08, energy, '↑', fontsize=12, verticalalignment='center')
                    ax.text(elec_pos-0.02, energy, '↓', fontsize=12, verticalalignment='center')
                elec_index += 1

        ax.set_ylabel("Energy")
        ax.set_xlim(-2, 2)
        ax.axhline(0, color='lightgray', linestyle='--', linewidth=1)
        ax.grid(False)
        ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    structure = {
        1: [2], 
        2: [1, 3], 
        3: [2, 4], 
        4: [3]
    }
    h = Huckel(structure)
    print(h.bond_orders)
    print(h.electron_density)
    print([v.evalf(4) for v in h.electron_density])
    h.plot_energy()
