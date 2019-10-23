# -*- coding: utf-8 -*-

import numpy as np
from .leaf import Leaf
from .branch import Branch


class RCTree(object):
    def __init__(self, X=None, index_labels=None, max_depth=10, precision=9, random_state=None):
        # Random number generation with provided seed
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random
        # Initialize dict for leaves
        self.leaves = {}
        # Initialize tree root
        self.root = None
        self.ndim = None
        self.max_depth = max_depth

        # Round data to avoid sorting errors
        X = np.around(X, decimals=precision)
        # Initialize index labels, if they exist
        if index_labels is None:
            index_labels = np.arange(X.shape[0], dtype=int)
        self.index_labels = index_labels
        # Check for duplicates
        U, I, N = np.unique(X, return_inverse=True, return_counts=True, axis=0)
        # If duplicates exist, take unique elements
        if N.max() > 1:
            n, d = U.shape
            X = U
        else:
            n, d = X.shape
            N = np.ones(n, dtype=np.int)
            I = None
        # Store dimension of dataset
        self.ndim = d
        # Set node above to None in case of bottom-up search
        self.u = None
        # Create RRC Tree
        S = np.ones(n, dtype=np.bool)
        self._mk_tree_(X, S, N, I, parent=self)
        # Remove parent of root
        self.root.u = None
        # Count all leaves under each branch
        self._count_all_top_down_(self.root)
        # Set bboxes of all branches
        self._get_bbox_top_down_(self.root)

    def __repr__(self):
        depth = ""
        treestr = ""

        def print_push(char):
            nonlocal depth
            branch_str = ' {}  '.format(char)
            depth += branch_str

        def print_pop():
            nonlocal depth
            depth = depth[:-4]

        def print_tree(node):
            nonlocal depth
            nonlocal treestr
            if isinstance(node, Leaf):
                treestr += '({})\n'.format(node.i)
            elif isinstance(node, Branch):
                treestr += '{0}{1}\n'.format(chr(9472), '+')
                treestr += '{0} {1}{2}{2}'.format(depth, chr(9500), chr(9472))
                print_push(chr(9474))
                print_tree(node.l)
                print_pop()
                treestr += '{0} {1}{2}{2}'.format(depth, chr(9492), chr(9472))
                print_push(' ')
                print_tree(node.r)
                print_pop()

        print_tree(self.root)
        return treestr

    def _cut_(self, X, S, parent=None, side='l'):
        # Find max and min over all d dimensions
        xmax = X[S].max(axis=0)
        xmin = X[S].min(axis=0)
        # Compute l
        l = xmax - xmin
        l /= l.sum()
        # Determine dimension to cut
        q = self.rng.choice(self.ndim, p=l)
        # Determine value for split
        p = self.rng.uniform(xmin[q], xmax[q])
        # Determine subset of points to left
        S1 = (X[:, q] <= p) & (S)
        # Determine subset of points to right
        S2 = (~S1) & (S)
        # Create new child node
        child = Branch(q=q, p=p, u=parent)
        # Link child node to parent
        if parent is not None:
            setattr(parent, side, child)
        return S1, S2, child

    def _mk_tree_(self, X, S, N, I, parent=None, side='root', depth=0):
        # Increment depth as we traverse down
        # if depth > self.max_depth:
        #     return
        depth += 1
        # Create a cut according to definition 1
        S1, S2, branch = self._cut_(X, S, parent=parent, side=side)
        # If S1 does not contain an isolated point...
        if S1.sum() > 1:
            # Recursively construct tree on S1
            self._mk_tree_(X, S1, N, I, parent=branch, side='l', depth=depth)
        # Otherwise...
        else:
            # Create a leaf node from isolated point
            i = np.ndarray.item(np.flatnonzero(S1))
            leaf = Leaf(i=i, d=depth, u=branch, x=X[i, :], n=N[i])
            # Link leaf node to parent
            branch.l = leaf
            # If duplicates exist...
            if I is not None:
                # Add a key in the leaves dict pointing to leaf for all duplicate indices
                J = np.flatnonzero(I == i)
                # Get index label
                J = self.index_labels[J]
                for j in J:
                    self.leaves[j] = leaf
            else:
                i = self.index_labels[i]
                self.leaves[i] = leaf
        # If S2 does not contain an isolated point...
        if S2.sum() > 1:
            # Recursively construct tree on S2
            self._mk_tree_(X, S2, N, I, parent=branch, side='r', depth=depth)
        # Otherwise...
        else:
            # Create a leaf node from isolated point
            i = np.ndarray.item(np.flatnonzero(S2))
            leaf = Leaf(i=i, d=depth, u=branch, x=X[i, :], n=N[i])
            # Link leaf node to parent
            branch.r = leaf
            # If duplicates exist...
            if I is not None:
                # Add a key in the leaves dict pointing to leaf for all duplicate indices
                J = np.flatnonzero(I == i)
                # Get index label
                J = self.index_labels[J]
                for j in J:
                    self.leaves[j] = leaf
            else:
                i = self.index_labels[i]
                self.leaves[i] = leaf
        # Decrement depth as we traverse back up
        depth -= 1

    def map_leaves(self, node, op=(lambda x: None), *args, **kwargs):
        if isinstance(node, Branch):
            if node.l:
                self.map_leaves(node.l, op=op, *args, **kwargs)
            if node.r:
                self.map_leaves(node.r, op=op, *args, **kwargs)
        else:
            op(node, *args, **kwargs)

    def map_branches(self, node, op=(lambda x: None), *args, **kwargs):
        if isinstance(node, Branch):
            if node.l:
                self.map_branches(node.l, op=op, *args, **kwargs)
            if node.r:
                self.map_branches(node.r, op=op, *args, **kwargs)
            op(node, *args, **kwargs)

    def forget_point(self, index):
        try:
            # Get leaf from leaves dict
            leaf = self.leaves[index]
        except KeyError:
            raise KeyError('Leaf must be a key to self.leaves')
        # If duplicate points exist...
        if leaf.n > 1:
            # Simply decrement the number of points in the leaf and for all branches above
            self._update_leaf_count_upwards_(leaf, inc=-1)
            return self.leaves.pop(index)
        # Weird cases here:
        # If leaf is the root...
        if leaf is self.root:
            self.root = None
            self.ndim = None
            return self.leaves.pop(index)
        # Find parent
        parent = leaf.u
        # Find sibling
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        # If parent is the root...
        if parent is self.root:
            # Delete parent
            del parent
            # Set sibling as new root
            sibling.u = None
            self.root = sibling
            # Update depths
            if isinstance(sibling, Leaf):
                sibling.d = 0
            else:
                self.map_leaves(sibling, op=self._increment_depth_, inc=-1)
            return self.leaves.pop(index)
        # Find grandparent
        grandparent = parent.u
        # Set parent of sibling to grandparent
        sibling.u = grandparent
        # Short-circuit grandparent to sibling
        if parent is grandparent.l:
            grandparent.l = sibling
        else:
            grandparent.r = sibling
        # Update depths
        parent = grandparent
        self.map_leaves(sibling, op=self._increment_depth_, inc=-1)
        # Update leaf counts under each branch
        self._update_leaf_count_upwards_(parent, inc=-1)
        # Update bounding boxes
        point = leaf.x
        self._relax_bbox_upwards_(parent, point)
        return self.leaves.pop(index)

    def _update_leaf_count_upwards_(self, node, inc=1):
        """
        Called after inserting or removing leaves. Updates the stored count of leaves
        beneath each branch (branch.n).
        """
        while node:
            node.n += inc
            node = node.u

    def insert_point(self, point, index, tolerance=None):
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if self.root is None:
            leaf = Leaf(x=point, i=index, d=0)
            self.root = leaf
            self.ndim = point.size
            self.leaves[index] = leaf
            return leaf
        # If leaves already exist in tree, check dimensions of point
        try:
            assert (point.size == self.ndim)
        except ValueError:
            raise ValueError(
                "Point must be same dimension as existing points in tree.")
        # Check for existing index in leaves dict
        try:
            assert (index not in self.leaves)
        except KeyError:
            raise KeyError("Index already exists in leaves dict.")
        # Check for duplicate points
        duplicate = self.find_duplicate(point, tolerance=tolerance)
        if duplicate:
            self._update_leaf_count_upwards_(duplicate, inc=1)
            self.leaves[index] = duplicate
            return duplicate
        # If tree has points and point is not a duplicate, continue with main algorithm...
        node = self.root
        parent = node.u
        maxdepth = max([leaf.d for leaf in self.leaves.values()])
        depth = 0
        branch = None
        for _ in range(maxdepth + 1):
            bbox = node.b
            cut_dimension, cut = self._insert_point_cut_(point, bbox)
            if cut <= bbox[0, cut_dimension]:
                leaf = Leaf(x=point, i=index, d=depth)
                branch = Branch(q=cut_dimension, p=cut, l=leaf, r=node,
                                n=(leaf.n + node.n))
                break
            elif cut >= bbox[-1, cut_dimension]:
                leaf = Leaf(x=point, i=index, d=depth)
                branch = Branch(q=cut_dimension, p=cut, l=node, r=leaf,
                                n=(leaf.n + node.n))
                break
            else:
                depth += 1
                if point[node.q] <= node.p:
                    parent = node
                    node = node.l
                    side = 'l'
                else:
                    parent = node
                    node = node.r
                    side = 'r'
        try:
            assert branch is not None
        except:
            raise AssertionError('Error with program logic: a cut was not found.')
        # Set parent of new leaf and old branch
        node.u = branch
        leaf.u = branch
        # Set parent of new branch
        branch.u = parent
        if parent is not None:
            # Set child of parent to new branch
            setattr(parent, side, branch)
        else:
            # If a new root was created, assign the attribute
            self.root = branch
        # Increment depths below branch
        self.map_leaves(branch, op=self._increment_depth_, inc=1)
        # Increment leaf count above branch
        self._update_leaf_count_upwards_(parent, inc=1)
        # Update bounding boxes
        self._tighten_bbox_upwards_(branch)
        # Add leaf to leaves dict
        self.leaves[index] = leaf
        # Return inserted leaf for convenience
        return leaf

    def query(self, point, node=None):
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if node is None:
            node = self.root
        return self._query_(point, node)

    def disp(self, leaf):
        if not isinstance(leaf, Leaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError(
                    'leaf must be a Leaf instance or key to self.leaves')
        # Handle case where leaf is root
        if leaf is self.root:
            return 0
        parent = leaf.u
        # Find sibling
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        # Count number of nodes in sibling subtree
        displacement = sibling.n
        return displacement

    def codisp(self, leaf):
        if not isinstance(leaf, Leaf):
            try:
                leaf = self.leaves[leaf]
            except KeyError:
                raise KeyError(
                    'leaf must be a Leaf instance or key to self.leaves')
        # Handle case where leaf is root
        if leaf is self.root:
            return 0
        node = leaf
        results = []
        for _ in range(node.d):
            parent = node.u
            if parent is None:
                break
            if node is parent.l:
                sibling = parent.r
            else:
                sibling = parent.l
            num_deleted = node.n
            displacement = sibling.n
            result = (displacement / num_deleted)
            results.append(result)
            node = parent
        co_displacement = max(results)
        return co_displacement

    def get_bbox(self, branch=None):
        if branch is None:
            branch = self.root
        mins = np.full(self.ndim, np.inf)
        maxes = np.full(self.ndim, -np.inf)
        self.map_leaves(branch, op=self._get_bbox_, mins=mins, maxes=maxes)
        bbox = np.vstack([mins, maxes])
        return bbox

    def find_duplicate(self, point, tolerance=None):
        nearest = self.query(point)
        if tolerance is None:
            if (nearest.x == point).all():
                return nearest
        else:
            if np.isclose(nearest.x, point, rtol=tolerance).all():
                return nearest
        return None

    def _lr_branch_bbox_(self, node):
        """
        Compute bbox of node based on bboxes of node's children.
        """
        bbox = np.vstack([np.minimum(node.l.b[0, :], node.r.b[0, :]),
                          np.maximum(node.l.b[-1, :], node.r.b[-1, :])])
        return bbox

    def _get_bbox_top_down_(self, node):
        """
        Recursively compute bboxes of all branches from root to leaves.
        """
        if isinstance(node, Branch):
            if node.l:
                self._get_bbox_top_down_(node.l)
            if node.r:
                self._get_bbox_top_down_(node.r)
            bbox = self._lr_branch_bbox_(node)
            node.b = bbox

    def _count_all_top_down_(self, node):
        """
        Recursively compute number of leaves below each branch from
        root to leaves.
        """
        if isinstance(node, Branch):
            if node.l:
                self._count_all_top_down_(node.l)
            if node.r:
                self._count_all_top_down_(node.r)
            node.n = node.l.n + node.r.n

    def _count_leaves_(self, node):
        """
        Count leaves underneath a single node.
        """
        num_leaves = np.array(0, dtype=np.int64)
        self.map_leaves(node, op=self._accumulate_, accumulator=num_leaves)
        num_leaves = np.ndarray.item(num_leaves)
        return num_leaves

    def _query_(self, point, node):
        """
        Recursively search for the nearest leaf to a given point.
        """
        if isinstance(node, Leaf):
            return node
        else:
            if point[node.q] <= node.p:
                return self._query_(point, node.l)
            else:
                return self._query_(point, node.r)

    def _increment_depth_(self, x, inc=1):
        """
        Primitive function for incrementing the depth attribute of a leaf.
        """
        x.d += (inc)

    def _accumulate_(self, x, accumulator):
        """
        Primitive function for helping to count the number of points in a subtree.
        """
        accumulator += (x.n)

    def _get_nodes_(self, x, stack):
        """
        Primitive function for listing all leaves in a subtree.
        """
        stack.append(x)

    def _get_bbox_(self, x, mins, maxes):
        """
        Primitive function for computing the bbox of a point.
        """
        lt = (x.x < mins)
        gt = (x.x > maxes)
        mins[lt] = x.x[lt]
        maxes[gt] = x.x[gt]

    def _tighten_bbox_upwards_(self, node):
        """
        Called when new point is inserted. Expands bbox of all nodes above new point
        if point is outside the existing bbox.
        """
        bbox = self._lr_branch_bbox_(node)
        node.b = bbox
        node = node.u
        while node:
            lt = (bbox[0, :] < node.b[0, :])
            gt = (bbox[-1, :] > node.b[-1, :])
            lt_any = lt.any()
            gt_any = gt.any()
            if lt_any or gt_any:
                if lt_any:
                    node.b[0, :][lt] = bbox[0, :][lt]
                if gt_any:
                    node.b[-1, :][gt] = bbox[-1, :][gt]
            else:
                break
            node = node.u

    def _relax_bbox_upwards_(self, node, point):
        """
        Called when point is deleted. Contracts bbox of all nodes above deleted point
        if the deleted point defined the boundary of the bbox.
        """
        while node:
            bbox = self._lr_branch_bbox_(node)
            if not ((node.b[0, :] == point) | (node.b[-1, :] == point)).any():
                break
            node.b[0, :] = bbox[0, :]
            node.b[-1, :] = bbox[-1, :]
            node = node.u

    def _insert_point_cut_(self, point, bbox):
        # Generate the bounding box
        bbox_hat = np.empty(bbox.shape)
        # Update the bounding box based on the internal point
        bbox_hat[0, :] = np.minimum(bbox[0, :], point)
        bbox_hat[-1, :] = np.maximum(bbox[-1, :], point)
        b_span = bbox_hat[-1, :] - bbox_hat[0, :]
        b_range = b_span.sum()
        r = self.rng.uniform(0, b_range)
        span_sum = np.cumsum(b_span)
        cut_dimension = np.inf
        for j in range(len(span_sum)):
            if span_sum[j] >= r:
                cut_dimension = j
                break
        if not np.isfinite(cut_dimension):
            raise ValueError("Cut dimension is not finite.")
        cut = bbox_hat[0, cut_dimension] + span_sum[cut_dimension] - r
        return cut_dimension, cut
