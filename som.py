# coding: utf-8

import numpy
from matplotlib import pylab as pl

class NamedVector():
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

class Cell():
    def __init__(self, cell_id=[], p, v, x):
        self.id = cell_id
        self.p = p
        self.v = v

        self.x = x
        self.xs = []

        self.n = 0
        self.d = 0

    def distanse(self, x, y):
        return np.linalg.norm(x - y)

    def diameter(self):
        if self.n != 0:
            return self.d / (self.n ** 2)
        return 0
    
    def absorbe(self, y):
        for x in self.xs:
            self.d += self.distanse(x.vector, y.vector)
            self.d += self.distanse(y.vector, x.vector)
        d += self.distanse(y.vector, y.vector)
        self.n += 1
        self.xs.append(y)
        return

    def is_dying(self):
        return False

    def is_critical(self):
        return False

    def kill(self):
        ret = self.xs
        self.xs = [] 
        self.n = 0
        self.d = 0
        return ret

    def new_id(self):
        return (self.id.append(0), self.id.append(1))

    def fission(self):
        return


class som():
    def __init__(self, vs):
        self.C = []
        self.N = 0
        self.t_boid = 1
        self.t_som = 1
        self.vs = vs

        self.v_dim = len(vs[0][1])

        self.e_boid = 100 #近傍の距離の定義
        self.e_som = 100 #近傍の距離の定義

    def push(self, x):
        if isinstance(x, list):
            self.vs.extend(x)
        else:
            self.vs.append(x)

    def ovum(self):
        self.C = [Cell(np.zeros(self.v_dim))]
        self.N = 1

    def arranged(self):
        return

    #Functions for Boid
    def d(self, xj, xi, e=None):
        if e is None:
            e = self.e_boid
        if np.linalg.norm(xj - xi) < e:
            return xj - xi
        return np.zeros(len(xj))

    def f_cohesion(self, v, t):
        return v / 100

    def cohesion(self, cell):
        p = -1 * cell.p
        for c in self.C:
            p += c.p
        return p / (self.N - 1)

    def f_alignment(self, v, t):
        return v / 8

    def alignment(self, cell):
        v = -1 * cell.v
        for c in self.C:
            v += c.v
        return v / (self.N - 1)

    def f_separation(self, v, t):
        return v

    def separation(self, cell):
        p = -1 * self.d(cell.p, cel.p)
        for c in self.C:
            p += self.d(c.p - cell.p)
        return p

    def f_organization(self, t):
        return

    def organization(self):
        return

    def update_boid(self, dt=1):
        new_v = []
        for c in self.C:
            new_v.append(c.v + self.f_cohesion(self.cohesion(c), self.t_boid)
                             + self.f_alignment(self.alignment(c), self.t_boid)
                             + self.f_separation(self.separation(c), self.t_boid)
                             #+ self.f_organization(self.organization(c), self.t)
                             )

        for c in self.C:
            c.v = new_v.pop(0)
            c.p += c.v * dt
        self.t_boid += 1
        return

    # Functions for SOM
    def get_matching_cell(self, x):
        cell = self.C[0]
        d = 0
        for c in self.C:
            d_buf = self.d(c.x, x)
            if d_buf < d:
                d = d_buf
                cell = c
        return cell

    # ユーザが選択してよい。ただしalpha(t)はt=0~1000くらいの間は1に近い値を示し、
    # その後単調減少するものである。
    def alpha(self, t):
        return 0.9(1 - t/1000)

    def sigma(self, t):
        return 100 / t

    def kernel(self, pc, pi, t):
        return self.alpha(t) * np.exp(-1 * (np.linalg.norm(pc - pi) ** 2) / (2 * (self.sigma(t) ** 2))

    def is_in_neighbor(self, pc, pi, e):
        if e is None:
            e = self.e_som
        if np.linalg.norm(pc - pi) < e:
            return True
        return False

    def update_som(self):
        if self.vs == []:
            return

        x = self.vs.pop(0)
        center = self.get_matching_cell(x.vector)
        for c in self.C:
            if self.is_in_neighbor(center.p, c.p)
                c.x += kernel(center.p, c.p, self.t_som) * (x.vector - c.x)
        self.t_som += 1
        return

    # Functions for Cell Behavior
    def assign(self, x):
        c = self.get_matching_cell(x.vector)
        c.absorbe(x)

    def reassign_all(self):
        v_buf = []
        for c in self.C:
            v_buf.extend(c.kill())
        for v in v_buf:
            self.assign(v)

    def has_critical_cell(self):
        for c in self.C:
            if c.is_dying() or c.is_critical():
                return True
        return False

    def metabolism(self):
        new_C = []
        corpse = []
        for c in self.C:
            if c.is_dying():
                corpse = c.kill() + corpse
                self.N -= 1
            else:
                if c.is_critical():
                    a, b = c.fission()
                    new_C.append(a)
                    new_C.append(b)
                    self.N += 1
                else:
                    new_C.append(c)
        self.C = new_C
        self.vs = corpse + self.vs
        return
    
    # Functions for integration
    def update(self, time_ratio=5):
        if self.vs == []:
            return
        for i in range(time_ratio):
            if self.vs == []:
                break
            self.update_som()
        while self.has_critical_cell():
            self.metabolism()
            self.reassign_all()
        self.update_boid()

    # Functions for visualization
    def show(self):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        for c in self.C:
            ax.scatter(c.p)
        pl.xlim([-1000, 1000])
        pl.title("SOM")

        pl.show()

    def animation(self):
        return
