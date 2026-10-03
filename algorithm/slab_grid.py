from .interfaces import np, plt, math
from .utils import get_color_for_s
from gurobi_tools.multiobjective import dominates, weakly_dominates
from .interfaces import TricksHypervolume as Tricks
from .numerics import obj_tol

class Grid:
    """
    Represents a 2D slab-based grid for hypervolume calculation.
    Is always grown from the ref point down to a certain coordinate. Does not necessarily go down to 0.
    Keys of slabs grow from left to right or from bottom to top.

    Keys, coordinates and areas are given, this class is only a struct to store the data. Use build_grid or build_grid_to_front to create instances.
    """
    def __init__(self, keys, coords, slab_width, direction:int):
        if direction not in [Tricks.VERT_SLABS, Tricks.HORIZ_SLABS]:
            raise ValueError("Grid direction must be either VERT_SLABS or HORIZ_SLABS")
        self.keys = list(sorted(keys)) # keys of coords-dict from left to right or from bottom to top
        self.coords = coords # dict of tuples. coord of bottom-left corner of each grid slab
        self.slab_width = slab_width # always called width, even if it is the height of the slab in case of horizontal slabs
        self.number_of_slabs = len(keys) # = 1 + maximal index in the keys - minimal index in the keys
        self.direction = direction

    def __repr__(self):
        return f"Grid(length={self.grid_length}, tiles x={self.tiles_in_x_direction}, tiles y={self.tiles_in_x_direction})"

def build_grid(ref_point, slab_width, direction) -> Grid:
    keys = []
    coords = {}

    anti_ref_point = (0,0)

    # n = floor(ref / slab_width), as in the paper (Sec. "Slab decomposition
    # and master model"). This guarantees the leftmost slab position
    # t^1 = ref - w*n lies in [0, w); using ceil instead would add an extra
    # slab with a negative left border that can never be activated.
    # The small epsilon guards the case of a mathematically integral ratio whose
    # floating-point quotient lands a hair below the integer (a lost slab).
    if direction == Tricks.VERT_SLABS:
        number_of_slabs = math.floor((ref_point[0] - anti_ref_point[0]) / slab_width + 1e-9)
    else:
        number_of_slabs = math.floor((ref_point[1] - anti_ref_point[1]) / slab_width + 1e-9)

    for i in range(number_of_slabs):
        if direction == Tricks.VERT_SLABS:
            x_pos = ref_point[0] - slab_width * (number_of_slabs - i)
            y_pos = anti_ref_point[1]
        else:
            x_pos = anti_ref_point[0]
            y_pos = ref_point[1] - slab_width * (number_of_slabs - i)
        coords[i] = (x_pos, y_pos)
        keys.append(i) # dont need to be sorted, will get sorted in Grid class

    return Grid(keys, coords, slab_width=slab_width, direction=direction)


def reduce_grid_to_fmin(full_grid:Grid, f_min, direction, one_overlap=False) -> Grid:
    """Keep the slabs of full_grid whose lower border is >= f_min, i.e. the slabs
    that some point with objective value >= f_min can activate.

    f_min comes out of a MILP and may carry solver noise. The comparison errs
    towards KEEPING a slab: an extra slab that no point can activate is harmless
    (its indicator stays 0 and h = 0), whereas dropping a slab that the true
    front activates makes the master underestimate the relaxation optimum and
    invalidates the gap certificate by that slab's area.
    """
    keys = []
    coords = {}
    f_min_tol = f_min - obj_tol(f_min)

    for f_key in full_grid.keys:
        test_coord = full_grid.coords[f_key][0] if direction == Tricks.VERT_SLABS else full_grid.coords[f_key][1]
        if one_overlap:
            test_coord += full_grid.slab_width
        if f_min_tol <= test_coord:
            coords[f_key] = full_grid.coords[f_key]
            keys.append(f_key)

    return Grid(keys, coords, full_grid.slab_width, full_grid.direction)


def plot_slabs_helper(ax, rect_orig, rect_width, rect_height, col, alpha, slab_width, sc_i, tot_sc):
    x_coor = rect_orig[0] + sc_i * slab_width
    while x_coor + slab_width <= rect_orig[0] + rect_width + slab_width/2:
        slab_x = [
            x_coor,
            x_coor + slab_width,
            x_coor + slab_width,
            x_coor
        ]
        slab_y = [
            rect_orig[1],
            rect_orig[1],
            rect_orig[1] + rect_height,
            rect_orig[1] + rect_height
        ]
        ax.fill(slab_x, slab_y, color=col, alpha=alpha, linewidth=0)
        x_coor = x_coor + slab_width * tot_sc

counter = -1
def plot_grid(axis, grid, D, slab_based_fronts, ref_point, annotate_cells=False):
    global counter
    counter += 1

    if grid.direction == Tricks.VERT_SLABS:
        grid_length_x = grid.slab_width
        grid_length_y = ref_point[1]
    else:
        grid_length_x = ref_point[0]
        grid_length_y = grid.slab_width


    for key in grid.keys:
        x, y = grid.coords[key]
        rect = plt.Rectangle((x, y), grid_length_x, grid_length_y, edgecolor='gray', facecolor='none', alpha=0.6)
        axis.add_patch(rect)
        if annotate_cells:
            axis.text(x, y, f"{key}", color='gray', fontsize=8, alpha=0.4)

    splits = 3
    slab_width = grid_length_x / len(D) / splits
    for si, s in enumerate(D):
        for fp in slab_based_fronts[s]:
            if grid.direction == Tricks.VERT_SLABS:
                plot_slabs_helper(axis, fp, grid_length_x, ref_point[1] - fp[1], get_color_for_s(si), 0.5, slab_width, si, len(D))
            else:
                plot_slabs_helper(axis, fp, ref_point[0] - fp[0], grid_length_y, get_color_for_s(si), 0.5, slab_width, si, len(D))

    axis.scatter([ref_point[0]], [ref_point[1]], label="Ref Point")

    xticks = np.arange(
        min(grid.coords[i][0] for i in grid.keys),
        max(grid.coords[i][0] for i in grid.keys) + grid_length_x,
        grid_length_x
    )
    xticklabels = [""] * len(xticks)
    for idx in range(1, len(xticks), max(1, int(len(xticks)/10))):
        x = xticks[idx]
        xticklabels[idx] = str(int(x))
    yticks = np.arange(
        min(grid.coords[i][1] for i in grid.keys),
        max(grid.coords[i][1] for i in grid.keys) + grid_length_y,
        grid_length_y
    )
    yticklabels = [""] * len(yticks)
    for idx in range(1, len(yticks), max(1, int(len(yticks)/10))):
        x = yticks[idx]
        yticklabels[idx] = str(int(x))
    axis.set_xticks(xticks)
    axis.set_xticklabels(xticklabels, rotation=90, ha='right')
    axis.set_yticks(yticks)
    axis.set_yticklabels(yticklabels, rotation=90, ha='right')

def setup_new_plot(plot_title):
    fig, ax = plt.subplots(dpi=400)
    ax.set_title(plot_title)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    return ax


def filter_front_to_ref(frontpoints_obj1, frontpoints_obj2, ref):
    """Keep only front points that weakly dominate the reference point, i.e.
    f_1 <= ref[0] and f_2 <= ref[1] (sort order in obj1 is preserved).

    epsilon_constraint_pareto_front does not know about the reference point, so
    a returned front may contain points outside the box [0,ref[0]] x [0,ref[1]].
    Points that do not dominate the reference point contribute no area to the
    hypervolume (they are removed before evaluating HV_LB/AV_LB in the paper,
    see Line "Find A_LB/HV_LB" of the algorithm); if left in, they would add
    negative area to the staircase/slab sums below. Filtering makes the
    HV/AV computations robust regardless of how the front was generated.
    """
    f1_out, f2_out = [], []
    for a, b in zip(frontpoints_obj1, frontpoints_obj2):
        if a <= ref[0] and b <= ref[1]:
            f1_out.append(a)
            f2_out.append(b)
    return f1_out, f2_out


def get_HV_from_front(frontpoints_obj1, frontpoints_obj2, ref_point):
    """
    Calculates the hypervolume from a given pareto front (assumed to be sorted ascending in obj1)
    :param frontpoints_obj1:
    :param frontpoints_obj2:
    :param ref_point:
    :return:
    """
    frontpoints_obj1, frontpoints_obj2 = filter_front_to_ref(frontpoints_obj1, frontpoints_obj2, ref_point)
    prevy = ref_point[1]
    ret = 0.0
    for fp_1, fp_2 in zip(frontpoints_obj1, frontpoints_obj2):
        ret += (prevy - fp_2) * (ref_point[0] - fp_1)
        prevy = fp_2
    return ret

def get_AV_vert_from_front(frontpoints_obj1, frontpoints_obj2, grid:Grid, ref):
    active_area = 0.0
    frontpoints_obj1, frontpoints_obj2 = filter_front_to_ref(frontpoints_obj1, frontpoints_obj2, ref)
    # given in ascending order of obj1
    frontpoints = list(zip(frontpoints_obj1, frontpoints_obj2))
    if not frontpoints:
        # no point dominates the reference point -> nothing is activated
        return 0.0
    fp = frontpoints.pop()
    for slab in reversed(grid.keys):
        while grid.coords[slab][0] < fp[0]:
            try:
                fp = frontpoints.pop()
            except:
                fp = (-np.inf, ref[1])
        active_area += (ref[1] - fp[1]) * grid.slab_width
    return active_area

def get_AV_horiz_from_front(frontpoints_obj1, frontpoints_obj2, grid:Grid, ref):
    active_area = 0.0
    frontpoints_obj1, frontpoints_obj2 = filter_front_to_ref(frontpoints_obj1, frontpoints_obj2, ref)
    # Mirror image of get_AV_vert_from_front with the two objectives swapped: a
    # horizontal slab with lower border t is activated by a point with f_2 <= t,
    # and among those the one with the LARGEST f_2 yields the widest extent
    # (ref_1 - f_1). The points arrive in ascending order of obj1, i.e. in
    # DESCENDING order of obj2, so we re-sort them ascending in obj2 and then
    # consume them with pop() from the largest f_2 downwards -- exactly as the
    # vertical routine consumes them from the largest f_1 downwards. (Consuming
    # the obj1-ascending list directly walks the front the wrong way and badly
    # underestimates the activated area.)
    frontpoints = sorted(zip(frontpoints_obj1, frontpoints_obj2), key=lambda p: p[1])
    if not frontpoints:
        # no point dominates the reference point -> nothing is activated
        return 0.0
    fp = frontpoints.pop()
    for slab in reversed(grid.keys):
        while grid.coords[slab][1] < fp[1]:
            try:
                fp = frontpoints.pop()
            except:
                fp = (ref[0], -np.inf)
        active_area += (ref[0] - fp[0]) * grid.slab_width
    return active_area

def get_AV_from_front(h, grid:Grid):
    active_area = 0.0
    for slab_index, h_var in h.items():
        active_area += h_var.X * grid.slab_width
    return active_area