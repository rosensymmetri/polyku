use crate::mask::{Dir, Mask, Node};
use rand::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    Free,
    Touched,
    Frozen,
}

impl Default for Color {
    fn default() -> Self {
        Self::Free
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rel {
    Strange,
    Child,
    Parent,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Rels {
    // We wrap this tiny array in a struct so that we can define what
    // it means for a 'Dir' to index into the array.
    rels: [Rel; 4],
}

impl std::ops::Index<Dir> for Rels {
    type Output = Rel;

    fn index(&self, d: Dir) -> &Self::Output {
        &self.rels[d.num()]
    }
}

impl std::ops::IndexMut<Dir> for Rels {
    fn index_mut(&mut self, d: Dir) -> &mut Self::Output {
        &mut self.rels[d.num()]
    }
}

impl std::default::Default for Rels {
    fn default() -> Self {
        Rels {
            rels: [Rel::Strange, Rel::Strange, Rel::Strange, Rel::Strange],
        }
    }
}

type Tree = Mask<Rels>;

impl Tree {
    fn contains(&self, n: Node) -> bool {
        let (nx, ny) = n.data();
        nx >= 0 && ny >= 0 && (nx as usize) < self.size() && (ny as usize) < self.size()
    }

    fn neighbors_dir(&self, n: Node) -> impl Iterator<Item = (Node, Dir)> + '_ {
        // One might expect 'neighbors' to return the nodes of the tree that
        // are related to the given node, but that is not the case. Rather,
        // it returns the nodes of the tree that are adjacent to the input node
        // and thus potentially are related to the input node. The direction
        // denotes which direction the node lies in from the input node.
        //
        // This is the primitive traversing function for the tree upon which
        // other methods are built.
        Dir::dirs()
            .map(move |d| (n + d, d))
            .filter(move |&(v, _)| self.contains(v))
    }

    fn neighbors_rel(&self, n: Node) -> impl Iterator<Item = (Node, Rel)> + '_ {
        self.neighbors_dir(n).map(move |(v, d)| (v, self[n][d]))
    }

    fn root(&self) -> Node {
        let mut root = Node::default();
        loop {
            if let Some(node) = self.parent(root) {
                root = node;
            } else {
                break;
            }
        }
        root
    }

    // This entire implementation block is for convenience methods for
    // navigating the tree.

    fn children(&self, n: Node) -> impl Iterator<Item = Node> + '_ {
        self.neighbors_rel(n)
            .filter(|&(_, r)| r == Rel::Child)
            .map(move |(v, _)| v)
    }

    fn parent(&self, n: Node) -> Option<Node> {
        self.neighbors_rel(n)
            .filter(|&(_, r)| r == Rel::Parent)
            .map(|(v, _)| v)
            .next()
    }
}

type Coloring = Mask<Color>;
type Tilings = Mask<Vec<usize>>;
type Partitions = Mask<Vec<Vec<usize>>>;

impl Coloring {
    fn free_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        self.iter_nodes()
            .filter(|&(_, &c)| c == Color::Free)
            .map(|(v, _)| v)
    }
}

struct TreeIter<'t> {
    next: Option<Node>,
    visited: Mask<bool>,
    tree: &'t Tree,
}

impl<'t> Iterator for TreeIter<'t> {
    type Item = Node;
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.next;
        loop {
            // This loop is broken when we either find a node whose children already
            // are visited, or we don't find a node. We first go to a leaf and then
            // start making our way up.

            if let Some(current) = self.next {
                self.visited[current] = true;
                if self.tree.children(current).all(|v| self.visited[v]) {
                    self.next = self.tree.parent(current);
                } else {
                    // This 'next' is not a None instance, because we have already
                    // established that not all its children are visited.
                    self.next = self
                        .tree
                        .children(current)
                        .filter(|&v| !self.visited[v])
                        .next();

                    break;
                }
            } else {
                break;
            }
        }

        ret
    }
}

impl Tree {
    fn iter_tree(&self) -> TreeIter {
        TreeIter {
            next: Some(self.root()),
            visited: <Mask<bool>>::new_default(self.size()),
            tree: self,
        }
    }
}

fn randomize_walk_from_tile_to_tree<R: Rng>(
    tree: &mut Tree,
    coloring: &mut Coloring,
    start: Node,
    rng: &mut R,
) {
    let mut path;
    'p: loop {
        path = Vec::new();
        let mut node = start;
        coloring[node] = Color::Touched;
        let mut dir: Dir;
        // We want to stop when we hit a frozen tile, in order to fuse
        // the tree with the new branch.
        while coloring[node] != Color::Frozen {
            // We avoid double crossing a branch and nodes with
            // too many relatives. We cannot necessarily deal with
            // nodes that have four relatives, and walking into a node
            // increases its number of relatives by one. Therefore we
            // require that it is strange with at least two nodes.
            let candidate = rand::seq::IteratorRandom::choose(
                tree.neighbors_dir(node).into_iter().filter(|&(v, _)| {
                    coloring[v] != Color::Touched
                        && tree
                            .neighbors_rel(v)
                            .filter(|&(_, r)| r == Rel::Child || r == Rel::Parent)
                            .count()
                            < 3
                }),
                rng,
            );

            match candidate {
                // Whenever we run into a situation where there are no
                // neighbors which haven't been touched yet, we restart
                // with a clean slate.
                None => {
                    for (v, _) in path.into_iter() {
                        coloring[v] = Color::Free;
                    }
                    continue 'p;
                }
                // We have a random selection and can continue building
                // this path.
                Some((v, d)) => {
                    node = v;
                    dir = d;
                    path.push((node, dir));
                    if coloring[node] == Color::Free {
                        // Mark the tiles that we have visited already as 'touched'.
                        coloring[node] = Color::Touched;
                    }
                }
            }
        }
        // After exiting the while scope, we have a path which
        // begins in 'tile' and ends in a tile which is 'frozen'.
        break 'p;
    }
    let mut prev = start;
    for (next, d) in path {
        // Mark that the previous node now belongs to the tree.
        coloring[prev] = Color::Frozen;
        // We will now mark the ancestry of the next and previous nodes.
        // Because we started at the leaf, the previous node is the child
        // of the next node, and therefore the previous node marks the next
        // node as 'Parent'.
        tree[prev][d] = Rel::Parent;
        tree[next][d.opp()] = Rel::Child;
        prev = next;
    }
    // Mission accomplished. No need for repitition.
}

fn fill_board_by_tree<R: rand::Rng>(size: usize, rng: &mut R) -> (Tree, Node) {
    println!("starting 'fill_board_by_tree'...");
    // Returns the relationship graph of a tree filling a board of given size,
    // together with the root of said tree.
    let mut tree = Tree::new_default(size);
    let root = Node::new(rng.gen_range(0, size as i32), rng.gen_range(0, size as i32));
    // 'coloring' keeps track of which tiles are occupied, and to which capacity.
    let mut occupancy = Coloring::new_default(size);
    occupancy[root] = Color::Frozen;

    while occupancy.free_nodes().count() > 0 {
        // Seeing as there is at least one free node left, we can randomly pick one.
        let next_leaf: Node = *occupancy
            .free_nodes()
            .collect::<Vec<Node>>()
            .choose(rng)
            .unwrap();
        // Fill in random branch from leaf to the tree.
        randomize_walk_from_tile_to_tree(&mut tree, &mut occupancy, next_leaf, rng);
    }
    (tree, root)
}

fn _partitions(parts: usize, n: usize) -> Vec<Vec<usize>> {
    let mut partitions = Vec::new();
    if parts == 0 {
        // The empty partition is a partition of n=0 into 0 parts.
        if n == 0 {
            partitions.push(Vec::new())
        }
        // If n>0 then it is not possible to partition n into 0 parts,
        // the set of partitions of n is the empty set.
        return partitions;
    }

    for i in 0..=n {
        for mut smaller_partition in _partitions(parts - 1, i).into_iter() {
            smaller_partition.push(n - i);
            partitions.push(smaller_partition);
        }
    }
    partitions
}

struct NumCycle<'a> {
    cursor: usize,
    tile_sizes: &'a [usize],
    exhausted: Vec<bool>,
    fresh: bool,
}

impl<'a> From<&'a Vec<usize>> for NumCycle<'a> {
    fn from(other: &'a Vec<usize>) -> Self {
        let len = other.len();
        let mut exh = Vec::new();

        for _ in 0..len {
            exh.push(false);
        }

        Self {
            cursor: 0,
            tile_sizes: other,
            exhausted: exh,
            fresh: true,
        }
    }
}

impl NumCycle<'_> {
    /// Marks the target 'idx' as exhausted. Panics if 'idx' is outside bounds.
    /// As one of the slots is filled, we mark 'self' as fresh.
    fn exhaust(&mut self, idx: usize) {
        self.exhausted[idx] = true;
        self.fresh = true;
        self.cursor = idx;
    }

    /// Marks all indices as not exhausted and 'self' as not fresh.
    fn unexhaust(&mut self) {
        for idx in 0..self.exhausted.len() {
            self.exhausted[idx] = false;
        }
        self.fresh = false;
    }

    /// Finds the first element to the right of 'self.cursor' that satisfies
    /// 'predicate' and is not exhausted yet. Note that it wraps around to the part
    /// before 'self.cursor'. If there is no such element but 'self' still is fresh,
    /// it retries by resetting/unexhausting all elements while marking 'self' as
    /// not fresh.
    fn take<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: Fn(usize) -> bool,
    {
        let fst = self
            .tile_sizes
            .iter()
            .enumerate()
            .skip(self.cursor)
            .filter(|&(i, &u)| predicate(u) && !self.exhausted[i])
            .chain(
                self.tile_sizes
                    .iter()
                    .enumerate()
                    .take(self.cursor)
                    .filter(|&(i, &u)| predicate(u) && !self.exhausted[i]),
            )
            .next();

        if let Some((idx, &num)) = fst {
            self.exhaust(idx);
            Some(num)
        } else if self.fresh {
            self.unexhaust();
            self.take(predicate)
        } else {
            None
        }
    }
}

/// Returns the tiling of the board.
fn subdivide_tree(root: Node, tree: &Tree, tasks: &Vec<usize>) -> Mask<usize> {
    println!("starting 'subdivide_tree'...");
    let mut coloring = Mask::<usize>::new_default(tree.size());
    // Keeps track of how many offspring a node is tasked with coloring.
    let mut tasked = Mask::<Option<usize>>::new_default(tree.size());
    // Keeps track of all the failed number of offspring a node was tasked
    // with coloring.
    let mut failed = Tilings::new_default(tree.size());
    // Keeps track of which children should carry the part of the number
    // of tiles assigned to this node.
    let mut partitions = Partitions::new_default(tree.size());
    // Keeps track of whether the children of this node are satisfied/done.
    // A leaf is satisfied when the number of nodes assigned by its parent
    // are zero.
    let mut satisfied = Mask::<bool>::new_default(tree.size());

    let mut tasks = NumCycle::from(tasks);

    // Let us take the first, best task availible.
    tasked[root] = tasks.take(|_| true);
    partitions[root] = _partitions(tree.children(root).count(), tasked[root].unwrap());

    coloring[root] = 0;

    let mut next = Some(root);
    let mut new_color = 1;

    // The essential strategy is that we color small connected regions of the tree
    // by starting at the root and working our way to the leaves. A node tasks its
    // children with painting some number of nodes in the same color as itself.
    //
    // This is done by generating a random partition of the number of nodes it needs
    // to paint. This can fail at a leaf, if it is tasked with painting more nodes
    // than itself. In that case it will backtrack to its parent, asking the parent
    // to try another partition. If a node runs out of options it will backtrack
    // further.
    while let Some(current) = next {
        if tree.children(current).any(|v| tasked[v] == None) {
            if let Some(partition) = partitions[current].pop() {
                // We get another partition to try with.
                for (&k, v) in partition.iter().zip(tree.children(current)) {
                    if tasked[v] == None {
                        if k > 0 {
                            tasked[v] = Some(k);
                            partitions[v] = _partitions(tree.children(v).count(), k - 1);
                            coloring[v] = coloring[current];
                        } else {
                            tasked[v] = tasks.take(|i| !failed[v].contains(&i));
                            match tasked[v] {
                                Some(task) => {
                                    partitions[v] = _partitions(tree.children(v).count(), task - 1);
                                }
                                None => break,
                            }
                            coloring[v] = new_color;
                            new_color += 1;
                        }
                    }
                }
            } else {
                // Having exhausted all the possible partitions of the current tilesize we
                // will have to try a new tilesize and partitions of said tilesize.
                // Note that 'current' node must already have been tasked with something.
                failed[current].push(tasked[current].unwrap());
                tasked[current] = None;
                next = tree.parent(current);
                if next.is_none() {
                    println!("{}", repr_tree_colors(&tree, &coloring));
                    panic!("failed to subdivide tree");
                }
            }
        } else if tree.children(current).all(|v| satisfied[v]) {
            satisfied[current] = true;
            next = tree.parent(current);
        } else {
            // 'next' has to be 'Some(_)' because there is some child 'v' such that
            // 'satisfied[v]' is false.
            next = tree.children(current).filter(|&v| !satisfied[v]).next();
        }
    }
    coloring
}

fn reset_local_tree(
    root: Node,
    tree: &Tree,
    satisfied: &mut Mask<bool>,
    tasked: &mut Mask<Option<usize>>,
    partitions: &mut Partitions,
) {
    // We reset everything but the root of the local tree.
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);

    loop {
        if let Some(current) = queue.pop_front() {
            // Clear the nodes that are not yet reset to initial state.
            // Note that a clear node always has clear children
            // so there is no need to traverse the entire subtree.
            for v in tree.children(current) {
                if tasked[v] != None {
                    satisfied[v] = false;
                    tasked[v] = None;
                    partitions[v].clear();
                    queue.push_back(v);
                }
            }
        } else {
            break;
        }
    }
}

#[derive(Default)]
pub(crate) struct Boundary {
    // The nodes are stored in counter-clockwise order. The 'length' allows for
    // indexing the boundary by any integer (not necessarily positive) by having
    // the index repeat every 'length'.
    length: usize,
    nodes: Vec<Node>,
}

impl std::ops::Index<usize> for Boundary {
    type Output = Node;
    fn index(&self, i: usize) -> &Node {
        &self.nodes[i % self.length]
    }
}

impl std::ops::IndexMut<usize> for Boundary {
    fn index_mut(&mut self, i: usize) -> &mut Node {
        &mut self.nodes[i % self.length]
    }
}

impl Boundary {
    fn contains(&self, n: &Node) -> bool {
        self.nodes.contains(n)
    }

    // Takes two different boundaries which have an inhabited and connected intersection,
    // produces the fusion of the two boundaries.
    fn fuse(&mut self, snd: Self) {
        // 'lower' is the index of the first node that is in both boundaries
        // If the nodes 0-2 have [0 in snd, 1 not in snd, 2 in snd, ...] we should get lower = 2.
        // This is because '0 in snd' is actually the last node in the intersection.
        let (fst_start, _) = self
            .nodes
            .iter()
            .enumerate()
            .cycle()
            .skip_while(|(_, v)| !snd.contains(v))
            .skip_while(|(_, v)| snd.contains(v))
            .next()
            .unwrap();
        let fst_len = self
            .nodes
            .iter()
            .cycle()
            .skip(fst_start)
            .take_while(|v| !snd.contains(v))
            .count()
            + 1;
        let fst_part: Vec<Node> = self
            .nodes
            .iter()
            .cloned()
            .cycle()
            .skip(fst_start)
            .take(fst_len)
            .collect();

        let (snd_start, _) = snd
            .nodes
            .iter()
            .enumerate()
            .cycle()
            .skip_while(|(_, v)| !self.contains(v))
            .skip_while(|(_, v)| self.contains(v))
            .next()
            .unwrap();
        let snd_len = snd
            .nodes
            .iter()
            .cycle()
            .skip(snd_start)
            .take_while(|v| !self.contains(v))
            .count()
            + 1;
        let snd_part: Vec<Node> = snd
            .nodes
            .iter()
            .cloned()
            .cycle()
            .skip(snd_start)
            .take(snd_len)
            .collect();

        let length = fst_part.len() + snd_part.len();
        let mut fusion = Vec::with_capacity(self.length + snd.length);
        fusion.extend(fst_part);
        fusion.extend(snd_part);

        // We update 'self' to be the fused boundary.
        *self = Self {
            length: length,
            nodes: fusion,
        };
    }

    fn from_node(n: Node) -> Self {
        let mut corners = Vec::with_capacity(4);
        corners.push(n);
        corners.push(n + Dir::S);
        corners.push(n + Dir::S + Dir::E);
        corners.push(n + Dir::E);
        Self {
            length: 4,
            nodes: corners,
        }
    }

    // We assume that 'coloring' is a mask on top of 'tree'.
    fn boundaries(tree: &Tree, coloring: &Mask<usize>) -> HashMap<usize, Self> {
        let mut boundaries: HashMap<usize, Self> = HashMap::new();

        for v in tree.iter_tree() {
            if let Some(boundary) = boundaries.get_mut(&coloring[v]) {
                (*boundary).fuse(Self::from_node(v));
            } else {
                boundaries.insert(coloring[v], Self::from_node(v));
            }
        }

        boundaries
    }

    pub(crate) fn data(self) -> Vec<Node> {
        self.nodes.into_iter().collect()
    }
}

fn random_slice<R: rand::Rng>(rng: &mut R) -> Vec<usize> {
    let mut vec = Vec::new();

    for _ in 0..2 {
        vec.push(1);
    }

    for _ in 0..10 {
        vec.push(2);
    }

    for _ in 0..10 {
        vec.push(3);
    }

    for _ in 0..1 {
        vec.push(4);
    }

    vec.shuffle(rng);
    vec
}

pub(crate) fn generate<R: rand::Rng>(
    size: usize,
    rng: &mut R,
) -> (
    Mask<usize>,               // the tiles corresponding to each 'color'
    HashMap<usize, Vec<Node>>, // the boundary vertices corresponding to each 'color'
) {
    let (tree, root) = fill_board_by_tree(size, rng);
    let coloring: Mask<usize> = subdivide_tree(root, &tree, &random_slice(rng));
    let boundaries = Boundary::boundaries(&tree, &coloring)
        .into_iter()
        .map(|(k, b)| (k, Boundary::data(b)))
        .collect();
    (coloring, boundaries)
}

struct DisplaySquare {
    // this vector has length 9
    chars: Vec<char>,
}

impl std::default::Default for DisplaySquare {
    fn default() -> Self {
        let mut chars = Vec::new();
        for _ in 0..15 {
            chars.push(' ')
        }
        DisplaySquare { chars }
    }
}

impl std::ops::Index<Node> for DisplaySquare {
    type Output = char;
    // The only valid indices are -1 <= x, y <= 1.
    fn index(&self, n: Node) -> &Self::Output {
        let (nx, ny) = n.data();
        &self.chars[(3 * (nx + 1) + ny + 1) as usize]
    }
}

impl std::ops::IndexMut<Node> for DisplaySquare {
    fn index_mut(&mut self, n: Node) -> &mut Self::Output {
        let (nx, ny) = n.data();
        &mut self.chars[(3 * (nx + 1) + ny + 1) as usize]
    }
}

impl DisplaySquare {
    fn line(&self, l: usize) -> String {
        // Crashes if l >= 3.
        let load = self
            .chars
            .chunks(3)
            .map(|chunk| chunk[l])
            .collect::<Vec<char>>();
        vec![load[0], ' ', load[1], ' ', load[2]]
            .into_iter()
            .collect::<String>()
    }
}

fn format_rel(d: Dir, r: Rel) -> char {
    match r {
        Rel::Strange => ' ',
        Rel::Child => match d {
            Dir::N | Dir::S => '|',
            Dir::W | Dir::E => '-',
        },
        Rel::Parent => match d {
            Dir::N => 'v',
            Dir::S => std::char::from_u32(0x2227).unwrap(), // wedge
            Dir::W => '>',
            Dir::E => '<',
        },
    }
}

fn format_node(tree: &Tree, n: Node, c: char) -> DisplaySquare {
    let o = Node::default();
    let mut sq = DisplaySquare::default();
    sq[o] = c;
    for (d, r) in Dir::dirs().map(|d| (d, tree[n][d])) {
        sq[o + d] = format_rel(d, r);
    }
    sq
}

fn repr_tree(tree: &Tree) -> String {
    let mut repr = String::new();
    for y in 0..tree.size() {
        // 'block' is the string corresponding to a row of nodes along the x axis
        let mut block = String::new();
        for l in 0..3 {
            for x in 0..tree.size() {
                block += &format_node(tree, Node::new(x as i32, y as i32), 'o').line(l);
                block += " ";
            }
            block += "\n";
        }
        repr += block.as_ref();
        repr += "\n";
    }
    repr
}

fn repr_tree_colors(tree: &Tree, tiling: &Mask<usize>) -> String {
    let mut repr = String::new();
    for y in 0..tree.size() {
        // 'block' is the string corresponding to a row of nodes along the x axis
        let mut block = String::new();
        for l in 0..3 {
            for x in 0..tree.size() {
                let n = Node::new(x as i32, y as i32);
                block += &format_node(
                    tree,
                    n,
                    std::char::from_digit((tiling[n] % 16) as u32, 16).unwrap(),
                )
                .line(l);
                block += " ";
            }
            block += "\n";
        }
        repr += block.as_ref();
        repr += "\n";
    }
    repr
}
