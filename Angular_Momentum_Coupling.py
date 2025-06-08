from fractions import Fraction
from functools import lru_cache
from collections import defaultdict
import numpy as np
from scipy.linalg import null_space


def solve_unit_kernel_vector(A):
    nullvecs = null_space(A)  # shape: (n, k)
    if nullvecs.shape[1] != 1: raise ValueError("Null space is not one-dimensional. Unique solution does not exist.")
    
    x = nullvecs[:, 0]
    if x[0] < 0:
        x = -x
    return x

def J_minus(j, m):
    k = (j+1-m) * (j+m)
    if k == 0:
        return (None, None, None)
    return (k**0.5, j, m-1)

def J_plus(j, m):
    k = (j+1+m) * (j-m)
    if k == 0:
        return (None, None, None)
    return (k**0.5, j, m+1)

def all_j(j1, j2):
    if 2*j1 != int(2*j1) or 2*j2 != int(2*j2):
        raise ValueError("j1 and j2 must be half-integers.")
    j_max = j1 + j2
    j_min = abs(j1 - j2)
    n = int(j_max - j_min + 1)
    j = [j_max - i for i in range(n)]
    return j

def all_m(j):
    if 2*j != int(2*j):
        raise ValueError("j must be a half-integer.")
    m_max = j
    m_min = -j
    n = int(m_max - m_min + 1)
    m = [m_max - i for i in range(n)]
    return m


@lru_cache()
def angular_momentum_coupling(j1, j2, j, m):
    if j1 < j2:
        return angular_momentum_coupling(j2, j1, j, m)
    if j < j1 - j2 or j > j1 + j2:
        return None
    if m < -j or m > j:
        return None
    if j == j1 + j2:
        if m == j:
            return {(j1, j2): 1}
        if m == -j:
            return {(-j1, -j2): 1}
    else:
        if m == j:
            n = j1 + j2 - j + 1
            if n != int(n):
                return None
            n = int(n)
            all_parts = [(j1-n+1+k, j2-k) for k in range(n)]
            A = np.array([[angular_momentum_coupling(j1, j2, j+i, m)[part] for part in all_parts] for i in range(1, n)])
            x = solve_unit_kernel_vector(A)
            res = {part: x[i] for i, part in enumerate(all_parts)}
            return res
        if m == -j:
            n = j1 + j2 - j + 1
            if n != int(n):
                return None
            n = int(n)
            all_parts = [(-(j1-n+1+k), k-j2) for k in range(n)]
            A = np.array([[angular_momentum_coupling(j1, j2, j+i, m)[part] for part in all_parts] for i in range(1, n)])
            x = solve_unit_kernel_vector(A)
            res = {part: x[i] for i, part in enumerate(all_parts)}
            return res
    if m >= 0:
        # 用J-作用于|j, m+1>得到|j, m>
        k0, *_ = J_minus(j, m+1)
        res = defaultdict(int)
        last = angular_momentum_coupling(j1, j2, j, m+1)
        if last is None:
            return None
        for (m1, m2), k in last.items():
            k1, *_ = J_minus(j1, m1)
            if k1 is not None:
                res[(m1-1, m2)] += k1/k0*k
            k2, *_ = J_minus(j2, m2)
            if k2 is not None:
                res[(m1, m2-1)] += k2/k0*k
        return dict(res)
    else:
        # 用J+作用于|j, m-1>得到|j, m>
        k0, *_ = J_plus(j, m-1)
        res = defaultdict(int)
        last = angular_momentum_coupling(j1, j2, j, m-1)
        if last is None:
            return None
        for (m1, m2), k in last.items():
            k1, *_ = J_plus(j1, m1)
            if k1 is not None:
                res[(m1+1, m2)] += k1/k0*k
            k2, *_ = J_plus(j2, m2)
            if k2 is not None:
                res[(m1, m2+1)] += k2/k0*k
        return dict(res)

def output(res: dict, keep_sqrt = False):
    if keep_sqrt:
        return ' + '.join([f"( sqrt({v**2:.4f}) |m1={Fraction(k[0])}, m2={Fraction(k[1])}> )" for k, v in res.items() if not np.isclose(v, 0, 1e-6)])
    else:
        return ' + '.join([f"( {v:.4f} |m1={Fraction(k[0])}, m2={Fraction(k[1])}> )" for k, v in res.items() if not np.isclose(v, 0, 1e-6)])
        
class Angular_Momentum_Coupling:

    def __init__(self, j1, j2):
        self.j1 = Fraction(j1)
        self.j2 = Fraction(j2)
        self._check()
    
    def _check(self):
        if 2*self.j1 != int(2*self.j1) or 2*self.j2 != int(2*self.j2):
            raise ValueError("j1 and j2 must be half-integers.")
        if self.j1 < 0 or self.j2 < 0:
            raise ValueError("j1 and j2 must be non-negative.")

    def angular_momentum_coupling(self, j, m):
        return angular_momentum_coupling(self.j1, self.j2, j, m)
    
    def output_all_couplings(self, keep_sqrt = False):
        for j in all_j(self.j1, self.j2):
            for m in all_m(j):
                res = self.angular_momentum_coupling(j, m)
                if res is not None:
                    print(f"|j={Fraction(j)}, m={Fraction(m)}> = {output(res, keep_sqrt)}")


if __name__ == "__main__":
    j1 = 5/2
    j2 = "1/2"
    # float, int, str 均可
    
    amc = Angular_Momentum_Coupling(j1, j2)
    amc.output_all_couplings()
    # |j=3, m=3> = ( 1.0000 |m1=5/2, m2=1/2> )
    # |j=3, m=2> = ( 0.9129 |m1=3/2, m2=1/2> ) + ( 0.4082 |m1=5/2, m2=-1/2> )
    # |j=3, m=1> = ( 0.8165 |m1=1/2, m2=1/2> ) + ( 0.5774 |m1=3/2, m2=-1/2> )
    # |j=3, m=0> = ( 0.7071 |m1=-1/2, m2=1/2> ) + ( 0.7071 |m1=1/2, m2=-1/2> )
    # |j=3, m=-1> = ( 0.8165 |m1=-1/2, m2=-1/2> ) + ( 0.5774 |m1=-3/2, m2=1/2> )
    # |j=3, m=-2> = ( 0.9129 |m1=-3/2, m2=-1/2> ) + ( 0.4082 |m1=-5/2, m2=1/2> )
    # |j=3, m=-3> = ( 1.0000 |m1=-5/2, m2=-1/2> )
    # |j=2, m=2> = ( 0.4082 |m1=3/2, m2=1/2> ) + ( -0.9129 |m1=5/2, m2=-1/2> )
    # |j=2, m=1> = ( 0.5774 |m1=1/2, m2=1/2> ) + ( -0.8165 |m1=3/2, m2=-1/2> )
    # |j=2, m=0> = ( 0.7071 |m1=-1/2, m2=1/2> ) + ( -0.7071 |m1=1/2, m2=-1/2> )
    # |j=2, m=-1> = ( 0.5774 |m1=-1/2, m2=-1/2> ) + ( -0.8165 |m1=-3/2, m2=1/2> )
    # |j=2, m=-2> = ( 0.4082 |m1=-3/2, m2=-1/2> ) + ( -0.9129 |m1=-5/2, m2=1/2> )
