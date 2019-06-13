import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.widgets import RadioButtons, AxesWidget, Button, TextBox
from sage.all import *
import pickle
import sys

# abelfunctions
from abelfunctions import *
from abelfunctions.complex_path import ComplexLine
from abelfunctions.utilities import Permutation, matching_permutation



def dist(p1, p2):
    ''' UTILITY method for euclidean distance'''
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def ccw(a, b, c):
    ''' UTILITY method for determining orientation '''
    return bool((c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]))

def intersect(a, b, c, d):
    ''' UTILITY method to check if two *line segments* intersect '''
    return bool(ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d))

def intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4)):
    ''' UTILITY method to find the intersection point of X1X2 and X3X3 '''
    px = float((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4))
    d = float((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    py = float((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4))
    return (px/d, py/d)


class BranchPoint:

    def __init__(self, sage_val, cp=None):
        self.sage_val = sage_val
        self.cp = cp
        self.is_finite = sage_val not in (+Infinity, -Infinity)
        self.val = sage_val.n() if self.is_finite else 1e10
        self.real = self.val.real() if self.is_finite else 1e10
        self.imag = self.val.imag() if self.is_finite else 0
        self.permutation = None
        self.branch_cut = None

    def _find_permutation_path(self, fineness=10):
        def _norm(x, scale=1, center=self.cp.cut_point):
            x = np.complex(x)
            r = np.complex(x - center)
            return np.complex(center + scale*r/np.abs(r))

        # radius of a circle encompassing all branch points
        R = 1.5*max(np.abs(x.val - self.cp.cut_point) for x in self.cp.branch_points if x.is_finite)
        # angle of this branch point
        self.angle = np.angle(self.val - self.cp.cut_point)

        base_point = _norm(self.cp.monodromy_point, scale=R)
        if self.is_finite:
            # half the distance to closest problematic point
            r = np.abs(np.complex(self.val - self.cp.cut_point))
            for x in self.cp.surface.discriminant_points:
                if x.n() != self.val:
                    r = min(r, 0.5*np.abs(np.complex(self.val - np.complex(x))))

            outter = _norm(self.val, scale=R)
            circle_start = _norm(self.val, scale=np.abs(self.val - self.cp.cut_point) + r)
            ang = np.angle(outter - self.cp.cut_point) - np.angle(base_point - self.cp.cut_point)
            if self.imag < self.cp.cut_point.imag():
                ang = 2*np.pi + ang

            points_to_outter = [self.cp.monodromy_point] + [np.complex(self.cp.cut_point) + np.complex(base_point-self.cp.cut_point)*np.exp(1j*ang*float(i)/fineness) for i in range(fineness+1)]
            around = [self.val + (circle_start-self.val)*np.exp(2j*np.pi*i/fineness) for i in range(fineness+1)]
            points = points_to_outter + around + points_to_outter[::-1]
        else:
            points = [self.cp.monodromy_point] + [np.complex(self.cp.cut_point) + np.complex(base_point-self.cp.cut_point)*np.exp(-2j*np.pi*i/fineness) for i in range(fineness+1)] + [self.cp.monodromy_point]

        # self.cp.ax.scatter([np.real(x) for x in points], [np.imag(x) for x in points])
        # return
        self.permutation_path = CyclePainterPath(points, 0, self.cp, color_sheets=False)
        self.permutation = matching_permutation(self.permutation_path.get_y(0), self.permutation_path.get_y(1))
        # print(self, self.permutation)

    def __str__(self):
        return '|BP|{' + str(self.sage_val) +', ' + str(self.permutation) + '} '

    def __repr__(self):
        return self.__str__()

    def show_permutation_path(self):
        self.permutation_path.display()


class CyclePainterPath:

    def __init__(self, projection_points, starting_sheet, cyclepainter, build_surface=True, color_sheets=True):
        projection_points = [np.complex(x) for x in projection_points]
        self.starting_sheet = starting_sheet
        self.projection_points = projection_points
        self.cp = cyclepainter

        if build_surface:
            path = ComplexLine(projection_points[0], projection_points[1])
            for i in range(1, len(projection_points)-1):
                path = path + ComplexLine(projection_points[i], projection_points[i+1])
                self._path = path
            self.surface_path = self.cp.surface._path_factory.RiemannSurfacePath_from_complex_path(path)
        else:
            self.surface_path = None

        segments = [[(np.real(x), np.imag(x)), (np.real(y), np.imag(y)) ] for x, y in zip(projection_points, projection_points[1:])]
        lines, sheets, colors = [], [], []

        # figure out the intersections
        current_sheet = starting_sheet
        for s in segments:
            intersections = []
            for x in self.cp.branch_points:
                if intersect(s[0], s[1], x.branch_cut[0], x.branch_cut[1]):
                    intersections.append((x, intersection(s[0], s[1], x.branch_cut[0], x.branch_cut[1])))
            intersections.sort(key=lambda (x, p): dist(s[0], p))

            last = s[0]
            for x, p in intersections:
                lines.append((last, p))
                sheets.append(current_sheet)
                colors.append(self.cp.sheet_color_map[current_sheet])

                clockwise = ccw((self.cp.cut_point.real(), self.cp.cut_point.imag()), p, s[1])
                if color_sheets:
                    current_sheet = x.permutation._list.index(current_sheet) if clockwise \
                                     else x.permutation._list[current_sheet]
                last = p
            lines.append((last, s[1]))
            sheets.append(current_sheet)
            colors.append(self.cp.sheet_color_map[current_sheet])

        self.lines = lines
        self.sheets = sheets
        self.line_collection = mc.LineCollection(lines, colors=colors, linewidths=2)

    def display(self, clear=True):
        if clear:
            self.cp.clear_canvas()
        self.cp.ax.add_collection(self.line_collection)
        self.cp.ax.autoscale()
        self.cp.fig.canvas.draw()

    def get_x(self, t):
        if not 0 <= t <= 1:
            print('The parameter must be in [0, 1].')
            return None
        return self.surface_path.get_x(t)

    def get_y(self, t):
        if not 0 <= t <= 1:
            print('The parameter must be in [0, 1].')
            return None
        return np.array(self.surface_path.get_y(t))

    def xy_path(self, n):
        # get n points equally spaced across the path
        return [(self.get_x(t), self.get_y(t)[self.starting_sheet]) for t in np.arange(0.0, 1.0 + 1.0/n, 1.0/n)]

    def is_closed(self, eps=1e-9):
        return dist((self.get_x(0), self.get_y(0)[self.starting_sheet]), (self.get_x(1), self.get_y(1)[self.starting_sheet])) < eps

    def integrate(self, omega):
        y0 = list(self.cp.surface.base_sheets)
        tmp = y0[self.starting_sheet]
        del y0[self.starting_sheet]
        y0 = np.array([tmp] + y0)

        sp = self.cp.surface._path_factory.RiemannSurfacePath_from_complex_path(self._path, y0=y0)
        return sp.integrate(omega)

    def intersection_number(self, path, eps=1e-11):
        ''' Calculates the intersection number with another path. '''
        intersections = 0
        for l, s in zip(self.lines, self.sheets):
            for ll, ss in zip(path.lines, path.sheets):
                if s == ss:
                    if intersect(l[0], l[1], ll[0], ll[1]):
                        intersection_point = intersection(l[0], l[1], ll[0], ll[1])
                        if min(dist(intersection_point, x) for x in [l[0], l[1], ll[0], ll[1]]) > eps:
                            intersections += 1 if ccw(l[0], intersection_point, ll[1]) else -1

        # around the monodromy point
        if path.sheets[0] == self.sheets[0]:
            e = (np.real(np.complex(self.get_x(1.-1e-9))), np.imag(np.complex(self.get_x(1.-1e-9))))
            s = (np.real(np.complex(self.get_x(1e-9))), np.imag(np.complex(self.get_x(1e-9))))
            ee = (np.real(np.complex(path.get_x(1.-1e-9))), np.imag(np.complex(path.get_x(1.-1e-9))))
            ss = (np.real(np.complex(path.get_x(1e-9))), np.imag(np.complex(path.get_x(1e-9))))
            if intersect(e, s, ee, ss):
                intersection_point = intersection(e, s, ee, ss)
                intersections += 1 if ccw(e, intersection_point, ss) else -1

        return intersections

    def apply_automorphism(self, f, fineness=200):
        image = [f(self.get_x(t), self.get_y(t)[self.starting_sheet])[0] for t in np.arange(0, 1.+1./fineness, 1./fineness)]
        self.cp.path_builder.start()
        for x in image:
            self.cp.path_builder.add(x)
        self.cp.path_builder.finish()


class PathBuilder:
    def __init__(self, cp):
        self.cp = cp
        self.points = []
        self.state = 'off'

    def start(self, event=None):
        self.cp.clear_canvas()
        self.points = [self.cp.monodromy_point]
        self.state = 'on'

    def finish(self, event=None):
        if self.state == 'on':
            self.points.append(self.cp.monodromy_point)
            self.state = 'off'
            self.display()

    def undo(self, event=None):
        if self.state == 'on':
            if len(self.points) > 1:
                self.points.pop()
                self.display()

    def pause(self, event=None):
        if self.state == 'on':
            self.state = 'pause'
        elif self.state == 'pause':
            self.state = 'on'

    def add(self, x):
        self.points.append(x)

    def click_listener(self, event=None):
        x = event.xdata, event.ydata
        if self.state == 'on':
            if event.inaxes == self.cp.ax:
                self.add(x[0] + I*x[1])
                self.display()

    def display(self):
        if len(self.points) > 1:
            CyclePainterPath(self.points, self.cp.radio_sheet, self.cp, build_surface=False).display()

    def _get_CyclePainterPath(self):
        return CyclePainterPath(self.points, self.cp.radio_sheet, self.cp)






class CyclePainter:

    def __init__(self, curve=None, initial_monodromy_point=None, cut_point=None, kappa=3./5.):
        #####################
        # mathematical
        #####################
        self.curve = curve # polynomial
        self.surface = RiemannSurface(curve, base_point=initial_monodromy_point, kappa=kappa) # RiemannSurface object
        self.degree = self.surface.degree # number of sheets
        self.kappa = kappa

        bp, _ = self.surface.monodromy_group()
        self.branch_points = [BranchPoint(x, cp=self) for x in bp]
        self.has_infinite_bp = any(not x.is_finite for x in self.branch_points)

        # Bounding box of the (finite) branch points
        self.real_span = max(x.real for x in self.branch_points if x.is_finite) - min(x.real for x in self.branch_points if x.is_finite)
        self.imag_span = max(x.imag for x in self.branch_points if x.is_finite) - min(x.imag for x in self.branch_points if x.is_finite)
        self.real_span = 1 if self.real_span == 0 else self.real_span
        self.imag_span = 1 if self.imag_span == 0 else self.imag_span

        #####################
        # other
        #####################
        self.radio_sheet = 0
        self.path_builder = PathBuilder(self)
        self.sheet_color_map = dict(enumerate(plt.cm.rainbow(np.linspace(0, 1, self.degree))))
        self.PATHS = {}

        #####################
        # computing values
        #####################
        # The center of CyclePainter cuts
        self.cut_point = cut_point if cut_point else self._find_cut_point()

        self.monodromy_point = np.complex(np.real(self.surface.base_point) + I*self.cut_point.imag())
        self.surface = RiemannSurface(curve, base_point=self.monodromy_point)

        bp, branch_permutations = self.surface.monodromy_group()
        self.branch_points = [BranchPoint(x, cp=self) for x in bp]

        # branch cuts
        _cut_point_coor = (self.cut_point.real(), self.cut_point.imag())
        _branch_cuts = [(_cut_point_coor, (x.real, x.imag)) for x in self.branch_points if x.is_finite]
        for x in self.branch_points:
            x.branch_cut = (_cut_point_coor, (x.real, x.imag))

        # self._compute_branch_permutations(branch_permutations)
        for x in self.branch_points:
            x._find_permutation_path()

    def _find_cut_point(self, fineness=8):
        '''
            Finds the point pc from the thesis.

            The heuristic to choose a suitable point will be:
                - avoid being too close to any branch point
                - sample possible candidates; we will use lattice of some fineness
                - pick the candidate which maximizes the minimal angle between
                    the candidate and two neighbouring branch points (when ordered
                    by angle from the candidate)
        '''
        finite_bp = [x for x in self.branch_points if x.is_finite]
        if len(finite_bp) == 0:
            return 0
        if len(finite_bp) == 1:
            return finite_bp[0].val + 1

        # Now we are dealing with at least two finite branch points.
        # As candidates, we will take some points close to centroid,
        # as well as a lattice of given fineness
        dreal, dimag = min(2, self.real_span/fineness), min(2, self.imag_span/fineness)
        d = min(dreal, dimag)
        # the centroid candidates
        centroid = sum(x.val for x in finite_bp) / len(finite_bp)
        candidates = [centroid + dx + dy*I for dx, dy in [(0,0), (d,0), (-d,0), (0,d), (0,-d)]]
        # the lattice candidates
        real_arange = np.arange(min(x.real for x in finite_bp) - dreal, max(x.real for x in finite_bp) + dreal, dreal)
        imag_arange = np.arange(min(x.imag for x in finite_bp) - dimag, max(x.imag for x in finite_bp) + dimag, dimag)
        candidates += [x + I*y for x in real_arange for y in imag_arange]

        max_min_angle, best_candidate = -Infinity, candidates[0]
        for cand in candidates:
            # discard candidates too close to branch points
            if min((cand - x.val).abs() for x in finite_bp) < d:
                continue

            # sort the angles of branch points from the candidate
            # (and add representations of +Infinity, if necessary)
            bp = [x.val for x in finite_bp]
            bp += [cand + 1] if self.has_infinite_bp else []
            bp += [self.surface.base_point]
            sbp = sorted(list(map(lambda p: atan2(-p.imag(), -p.real()).n(), [(p - cand).n() for p in bp])))
            # find the minimal angle
            min_angle = min(abs((sbp[i] - sbp[(i+1)%len(sbp)]) % float(2*pi)) for i in range(len(sbp)))

            if min_angle > max_min_angle:
                max_min_angle, best_candidate = min_angle, cand.n()

        return best_candidate

    def _radio_handler(self, label):
        self.radio_sheet = int(label)
        self.path_builder.display()

    def draw_basics(self):
        # draw monodromy point
        self.ax.scatter([float(np.real(self.monodromy_point))], [float(np.imag(self.monodromy_point))], c='k', marker='*', zorder=3)

        # draw cut point
        self.ax.scatter([self.cut_point.real()], [self.cut_point.imag()], c='k', marker='.', zorder=3)

        # draw branch points
        self.ax.scatter([x.real for x in self.branch_points if x.is_finite],
                        [x.imag for x in self.branch_points if x.is_finite],
                        c='k', marker='x', zorder=3)

        # draw kappa circles around branch points
        for x in self.branch_points:
            if x.is_finite:
                self.ax.add_artist(plt.Circle((x.real, x.imag), self.kappa, color='k', alpha=0.06, linestyle=':'))

        # draw discriminant points
        self.ax.scatter([x.real() for x in self.surface.discriminant_points],
                        [x.imag() for x in self.surface.discriminant_points],
                        c='k', marker='.', zorder=3)

        # annotate all of the branch points
        d_ann = (min(self.imag_span, self.real_span)/100).n()
        for i, x in enumerate(self.branch_points):
            annotation = r'({:d})'.format(i)
            self.ax.annotate(annotation, (x.real+d_ann, x.imag+d_ann), fontsize=7)
            if not x.is_finite:
                self.ax.annotate(r'$\infty$', (max(x.real for x in self.branch_points if x.is_finite) + self.real_span/6+0.01,
                                                self.cut_point.imag()-0.05), fontsize=9)

        # branch cuts
        _cut_point_coor = (self.cut_point.real(), self.cut_point.imag())
        _branch_cuts = [(_cut_point_coor, (x.real, x.imag)) for x in self.branch_points if x.is_finite]
        for x in self.branch_points:
            x.branch_cut = (_cut_point_coor, (x.real, x.imag))
        if self.has_infinite_bp:
            _branch_cuts.append((_cut_point_coor, (max(x.real for x in self.branch_points if x.is_finite) + self.real_span/6, self.cut_point.imag()) ))
        branch_cuts = mc.LineCollection(_branch_cuts, colors='k', linewidths=0.4)

        # main
        self.ax.add_collection(branch_cuts)
        self.ax.autoscale()
        self.ax.margins(0.1)
        self.ax.set_xlabel('Re')
        self.ax.xaxis.set_label_coords(1.05, 0)
        self.ax.set_ylabel('Im')
        self.ax.yaxis.set_label_coords(0, 1.05)
        self.ax.set_title('CyclePainter 2: Auckland', family='cursive')



        # add legend for the colors of sheets
        # first, some dummy plotting is necessary to get the legend
        for sheet_number, sheet_color in self.sheet_color_map.items():
            self.ax.plot([0,0], [0,0], c=sheet_color, label=str(sheet_number))

         # add the legend and adjust the plot
        legend = self.ax.legend(loc='best', framealpha=0, bbox_to_anchor=(1,-0.05), ncol=self.degree, handletextpad=-0.7, fontsize=7)
        for l in legend.legendHandles:
            l.set_linewidth(8)
        for text in legend.get_texts():
            text.set_color("white")
            text.set_weight("bold")
        plt.subplots_adjust(bottom=0.2)
        plt.subplots_adjust(left=0.2)

    def clear_canvas(self):
        self.ax.cla()
        self.draw_basics()

    def start(self):
        # print information first
        print('Curve:\n    ' + str(self.curve) + '\n')
        print('Kappa:\n    ' + str(self.kappa) + '\n')
        print('Monodromy point:\n      ' + str(self.monodromy_point) + '\n')
        print('Ordering of sheets at the monodromy point:')
        for i, x in enumerate(self.surface.base_sheets):
            print('   ({:d}) {:s}'.format(i, str(x)))
        print('Discriminant points:')
        for x in self.surface.discriminant_points:
            print('     ' + str(x))
        print('\nBranch cut point:\n      ' + str(self.cut_point) + '\n')
        for i, x in enumerate(self.branch_points):
            print('Branch point ({:d})'.format(i))
            print('     Value: {:s}'.format(str(x.sage_val)))
            print('     Permutation: {:s}'.format(str(x.permutation).replace(', (', '(').replace(',', '')[1:-1]))


        # open up a new plot
        self.fig, self.ax = plt.subplots(dpi=140)

        # draw all basics
        self.draw_basics()

        # connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.path_builder.click_listener)

        # add radio buttons for the starting sheet
        radioax = plt.axes([0.7, 0.55, 0.6, 0.03*self.degree], frameon=False, aspect='equal')
        self.radio = RadioButtons(radioax, map(str, range(self.degree)), activecolor='black')
        for circle in self.radio.circles: # adjust radius here. The default is 0.05
            circle.set_radius(0.4/self.degree)
        self.radio.on_clicked(self._radio_handler)

        # add buttons
        ax_start = plt.axes([0.02, 0.85, 0.08, 0.04])
        self.draw_path_button = Button(ax_start, 'New path', color=(0,0.7,0))
        self.draw_path_button.on_clicked(self.path_builder.start)
        self.draw_path_button.label.set_fontsize(6)

        ax_save = plt.axes([0.02, 0.8, 0.08, 0.04])
        self.save_button = Button(ax_save,'Finish path', color=(0,0.5,0.9))
        self.save_button.on_clicked(self.path_builder.finish)
        self.save_button.label.set_fontsize(6)

        ax_back = plt.axes([0.02, 0.75, 0.08, 0.04])
        self.undo_button = Button(ax_back,'Undo')
        self.undo_button.on_clicked(self.path_builder.undo)
        self.undo_button.label.set_fontsize(6)

        ax_pause = plt.axes([0.02, 0.7, 0.08, 0.04])
        self.pause_button = Button(ax_pause,'(Un)Pause')
        self.pause_button.on_clicked(self.path_builder.pause)
        self.pause_button.label.set_fontsize(6)

    def save_path(self, path_name):
        if path_name in self.PATHS:
            print('Fail: The path with name "{:s}" already exists.'.format(path_name))
            return
        if self.path_builder.state != 'off':
            print('Fail: The currently built path is not finished.')
            return
        p = self.path_builder._get_CyclePainterPath()
        if not p.is_closed():
            print('Warning: The path is not closed.')
        self.PATHS[path_name] = p
        print('Success: The path with name "{:s}" has been saved.'.format(path_name))
        return p

    def delete_path(self, path_name):
        if not path_name in self.PATHS:
            print('Fail: The path with name "{:s}" does not exists.'.format(path_name))
            return
        del self.PATHS[path_name]
        print('Success: The path with name "{:s}" has been deleted.'.format(path_name))

    def get_path(self, path_name):
        if not path_name in self.PATHS:
            print('Fail: The path with name "{:s}" does not exist.'.format(path_name))
            return
        return self.PATHS[path_name]

    def plot_path(self, path_name):
        if not path_name in self.PATHS:
            print('Fail: The path with name "{:s}" does not exist.'.format(path_name))
            return
        self.PATHS[path_name].display()

    def saved_paths(self):
        print('Saved paths:')
        for k in self.PATHS:
            print('    ' + k)

    def pickle_paths(self, filename):
        d = {name: (self.PATHS[name].projection_points, self.PATHS[name].starting_sheet) for name in self.PATHS}
        with open(filename, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_paths(self, filename):
        with open(filename, 'rb') as handle:
            d = pickle.load(handle)
        for name in d:
            self.PATHS[name] = CyclePainterPath(d[name][0], d[name][1], self)

    def period_matrix(self, a_cycle_names, b_cycle_names, differentials):
        return np.matrix([[self.PATHS[name].integrate(d) for name in (a_cycle_names+b_cycle_names)] for d in differentials])

    def riemann_matrix(self, a_cycle_names, b_cycle_names, differentials):
        pm = self.period_matrix(a_cycle_names, b_cycle_names, differentials)
        A, B = pm[:,:3], pm[:,3:]
        return np.matmul(np.linalg.inv(A), B)

    def intersection_matrix(self, path_names):
        return np.matrix([[self.PATHS[x].intersection_number(self.PATHS[y]) for y in path_names] for x in path_names])
