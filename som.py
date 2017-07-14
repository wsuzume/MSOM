# coding: utf-8

import random
import numpy as np
from matplotlib import pylab as pl
from matplotlib import animation


def update_plot(i, msom, im):
    pl.cla()
    print('frame:', i)
    msom.update_boid(0.1)
    for c in msom.C:
        pl.scatter(c.p[0], c.p[1], c='b')
    #pl.xlim([-1000, 1000])
    pl.title('time={0}'.format(i))


class NamedVector():
    def __init__(self, name, vector):
        self.name = name
        self.vector = vector

    def dump(self):
        print('[', self.name, ',', self.vector, ']')

class Cell():
    def __init__(self, cell_id, p, v, x):
        self.cell_id = cell_id
        self.p = p
        self.v = v

        self.x = x
        self.xs = []

        self.n = 0
        self.d = 0

        self.critical_diameter = 50

    def distance(self, x, y):
        return np.linalg.norm(x - y)

    def diameter(self):
        if self.n != 0:
            return self.d / (self.n ** 2)
        return 0
    
    def absorbe(self, y):
        #print(self.xs, y)
        if y in self.xs:
            #print('full')
            return
        #print('eat')
        for x in self.xs:
            self.d += self.distance(x.vector, y.vector)
            self.d += self.distance(y.vector, x.vector)
        self.d += self.distance(y.vector, y.vector)
        self.n += 1
        self.xs.append(y)
        return

    def is_dying(self):
        if self.n < 3:
            return True
        return False

    def is_critical(self):
        if self.diameter() > self.critical_diameter:
            return True
        return False

    def is_same_kind(self, c, e=1):
        if np.linalg.norm(c.x - self.x) < e:
            return True
        return False

    def kill(self):
        ret = self.xs
        self.xs = [] 
        self.n = 0
        self.d = 0
        return ret

    def new_id(self):
        return (self.cell_id + [0], self.cell_id + [1])

    def fission(self):
        id_a, id_b = self.new_id()
        a = Cell(id_a, self.p + np.random.normal(0, 0.4, 2),
                       self.v + np.random.normal(0, 0.1, 2),
                       self.x + np.random.normal(0, 0.1, len(self.x)))
        b = Cell(id_b, self.p + np.random.normal(0, 0.4, 2),
                       self.v + np.random.normal(0, 0.1, 2),
                       self.x + np.random.normal(0, 0.1, len(self.x)))
        for x in self.xs:
            d_a = self.distance(a.x, x.vector)
            d_b = self.distance(b.x, x.vector)
            if d_a < d_b:
                a.absorbe(x)
            else:
                b.absorbe(x)
        return (a, b)

    def fusion(self):
        return

    def dump(self):
        print('Cell:', self.cell_id)
        for x in self.xs:
            x.dump()

class SOM():
    def __init__(self, vs):
        self.C = []
        self.N = 0
        self.t_boid = 1
        self.t_som = 1
        self.vs = vs

        self.v_dim = len(vs[0].vector)

        self.e_boid = 0.5 #写像した2次元平面における近傍の距離の定義
        self.e_som = 100 #写像前の高次元ベクトルにおける近傍の距離の定義

        self.cohesion_const = 100
        self.alignment_const = 50
        self.separation_const = 10
        self.organization_const = 0
        self.heat_disturbance_const = 500

    def push(self, x):
        if isinstance(x, list):
            self.vs.extend(x)
        else:
            self.vs.append(x)

    def ovum(self):
        self.C = [Cell([], np.array([0, 0]), np.array([0, 0]), np.zeros(self.v_dim))]
        self.N = 1

    def fission(self):
        self.ovum()
        for i in range(5):
            new_C = []
            for c in self.C:
                a, b = c.fission()
                new_C.append(a)
                new_C.append(b)
                self.N += 1
            self.C = new_C

    def arranged(self, scale=1):
        v1 = np.array([1, 0])
        v2 = np.array([np.cos(- 1 * np.pi / 3), np.sin(-1 * np.pi / 3)])
        startpos1 = [
            -1 * v1 + -2 * v2,
            -2 * v1,
            -3 * v1 +  2 * v2
        ]
        startpos2 = [
            -1 * v1 + -1 * v2,
            -2 * v1 +  1 * v2
        ]

        cell_id = 0
        for p in startpos1:
            for i in range(5):
                self.C.append(Cell(cell_id, p + i * v1, np.zeros(2), np.random.random(self.v_dim)))
                self.N += 1
                cell_id += 1
        for p in startpos2:
            for i in range(4):
                self.C.append(Cell(cell_id, p + i * v1, np.zeros(2), np.random.random(self.v_dim)))
                self.N += 1
                cell_id += 1
        return

    #Functions for Boid
    def d(self, xj, xi, e=None):
        if e is None:
            e = self.e_boid
        if np.linalg.norm(xi - xj) < e:
            return xi - xj
        return np.zeros(len(xj))

    def f_cohesion(self, v, t):
        norm = np.linalg.norm(v)
        if norm > 1.0:
            v = v / norm
        return v * self.cohesion_const

    def cohesion(self, cell):
        if self.N <= 1:
            return np.zeros(len(cell.p))
        p = -1 * cell.p
        for c in self.C:
            p = p + c.p
        return p / (self.N - 1)

    def f_alignment(self, v, t):
        norm = np.linalg.norm(v)
        if norm > 1.0:
            v = v / norm
        return v * self.alignment_const

    def alignment(self, cell):
        if self.N <= 1:
            return np.zeros(len(cell.p))
        v = -1 * cell.v
        for c in self.C:
            v = v + c.v
        return v / (self.N - 1)

    def f_separation(self, v, t):
        norm = np.linalg.norm(v)
        if norm < 1e-3:
            norm = 1e-3
        v = v / (norm ** 2)
        return v * self.separation_const

    def separation(self, cell):
        if self.N <= 1:
            return np.zeros(len(cell.p))
        p = -1 * self.d(cell.p, cell.p)
        for c in self.C:
            p = p + self.d(c.p, cell.p)
        return p

    def f_organization(self, v, t):
        return

    def organization(self):
        return

    def f_heat_disturbance(self, v, t):
        return self.heat_disturbance_const * v / t

    def heat_disturbance(self, cell):
        return np.random.normal(0, 1, len(cell.p))

    def center_of_gravity(self):
        p = np.zeros(len(self.C[0].p))
        if self.N == 0:
            return p
        for c in self.C:
            p = p + c.p
        return p / self.N

    def update_boid(self, dt=1):
        new_v = []
        for c in self.C:
            v1 = self.f_cohesion(self.cohesion(c), self.t_boid)
            v2 = self.f_alignment(self.alignment(c), self.t_boid)
            v3 = self.f_separation(self.separation(c), self.t_boid)
            #v5 = self.f_heat_disturbance(self.heat_disturbance(c), self.t_boid)
            new_v.append(c.v + v1 + v2 + v3)
            #print(v1, v2, v3)
                             #+ self.f_organization(self.organization(c), self.t)
            #                 )
            #new_v.append(c.v + self.f_cohesion(self.cohesion(c), self.t_boid)
            #                 + self.f_alignment(self.alignment(c), self.t_boid)
            #                 + self.f_separation(self.separation(c), self.t_boid)
                             #+ self.f_organization(self.organization(c), self.t)
            #                 )

        for c in self.C:
            c.v = new_v.pop(0)
            vnorm = np.linalg.norm(c.v)
            if vnorm > 1.0:
                c.v = c.v / vnorm
            c.p = c.p + c.v * dt
        g = self.center_of_gravity()
        for c in self.C:
            c.p = c.p - g
        self.t_boid += 1
        return

    # Functions for SOM
    def get_matching_cell(self, x):
        cell = self.C[0]
        d = cell.distance(cell.x, x)
        for c in self.C[1:]:
            d_buf = cell.distance(c.x, x)
            if d_buf < d:
                d = d_buf
                cell = c
        return cell

    # ユーザが選択してよい。ただしalpha(t)はt=0~1000くらいの間は1に近い値を示し、
    # その後単調減少するものである。
    def alpha(self, t):
        if t < 1000:
            return 0.9
        return 900 / t

    def sigma(self, t):
        return 100 / t

    def kernel(self, pc, pi, t):
        return self.alpha(t) * np.exp(-1 * (np.linalg.norm(pc - pi) ** 2) / (2 * (self.sigma(t) ** 2)))

    def is_in_neighbor(self, pc, pi, e=3.0):
        if e is None:
            e = self.e_som
        if np.linalg.norm(pc - pi) < e:
            return True
        return False

    def update_som(self):
        if self.vs == []:
            return False

        x = self.vs.pop(0)
        center = self.get_matching_cell(x.vector)
        for c in self.C:
            if self.is_in_neighbor(center.p, c.p):
                c.x = c.x + self.kernel(center.p, c.p, self.t_som) * (x.vector - c.x)
        self.t_som += 1
        return True

    def update_som_assign(self):
        if self.vs == []:
            return False

        x = self.vs.pop(0)
        center = self.get_matching_cell(x.vector)
        center.absorbe(x)
        for c in self.C:
            if self.is_in_neighbor(center.p, c.p):
                c.x = c.x + self.kernel(center.p, c.p, self.t_som) * (x.vector - c.x)
        self.t_som += 1
        return True

    def update_som_repeatedly(self, n=1):
        if self.vs == []:
            return

        for i in range(n):
            print('phase:', i)
            random.shuffle(self.vs)
            for x in self.vs:
                center = self.get_matching_cell(x.vector)
                for c in self.C:
                    if self.is_in_neighbor(center.p, c.p):
                        c.x = c.x + self.kernel(center.p, c.p, self.t_som) * (x.vector - c.x)
                self.t_som += 1
        return

    def clustering(self):
        for v in self.vs:
            self.assign(v)

    # Functions for Cell Behavior
    def assign(self, x):
        c = self.get_matching_cell(x.vector)
        c.absorbe(x)

    def reload(self):
        for c in self.C:
            self.vs.extend(c.xs)
        random.shuffle(self.vs)

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
        for c in self.C:
            if c.is_critical():
                a, b = c.fission()
                new_C.append(a)
                new_C.append(b)
                self.N += 1
            else:
                new_C.append(c)
        self.C = new_C
        return

    def apoptosis(self):
        new_C = []
        corpse = []
        for c in self.C:
            if c.is_dying():
                corpse = c.kill() + corpse
                self.N -= 1
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
    def dump(self):
        for c in self.C:
            c.dump()

    def show(self):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        for c in self.C:
            ax.scatter(c.p[0], c.p[1], c='b')
        #pl.xlim([-1000, 1000])
        pl.title("SOM")

        pl.show()


    def animation(self):
        fig = pl.figure()

        ani = animation.FuncAnimation(fig, update_plot, fargs=(self, []), interval=100, frames=200)
        ani.save('sample.gif', writer='imagemagick')
        pl.show()
        return

    def plot_cells(self, t):
        xs = []
        ys = []
        for c in self.C:
            xs.append(c.p[0])
            ys.append(c.p[1])
        #pl.title('time={0}'.format(t))
        return pl.scatter(xs, ys, c='b')

    def execute_with_animation(self, path, time_ratio=20):
        if self.vs == []:
            return

        fig = pl.figure()
        ims = []
        self.ovum()
        ims.append([self.plot_cells('init')])
        for t in range(500):
            print('time=', t)
            for c in self.C:
                print('Cell:', c.cell_id, 'pos=', c.p, 'size=', len(c.xs), 'diameter=', c.diameter())
            for i in range(time_ratio):
                if self.vs == []:
                    self.reload()
                self.update_som_assign()
            if self.has_critical_cell():
                print('metabolism occured')
                self.metabolism()
                self.reassign_all()
                self.apoptosis()
            self.update_boid()
            ims.append([self.plot_cells(t)])
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save("output.gif", writer="imagemagick")
        self.reassign_all()
        self.apoptosis()
        self.write(path)

    def write(self, path):
        with open(path, 'w') as f:
            f.write('Cell num: {0}\n'.format(len(self.C)))
            for c in self.C:
                f.write('Cell: {0}\n'.format(c.cell_id))
                for x in c.xs:
                    f.write('\t{0}\n'.format(x.name))
