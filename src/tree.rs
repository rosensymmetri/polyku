use rand::prelude::*;
use std::collections::HashMap;
use crate::mask::{Node,Dir};

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
    rels: [Rel; 4]
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
        Rels{ rels: [Rel::Strange, Rel::Strange, Rel::Strange, Rel::Strange] }        
    }
}

struct Tree {
    // Carries the information about the graph concerning how it is connected,
    // which node is the parent/child of which node.
    size: usize,
    rels: Vec<Rels>,
}

impl Tree {
    fn new(size: usize) -> Self {
        // Initially none of the tiles are related.
        let mut r = Vec::with_capacity(size*size);
        for _ in 0..size*size {
            r.push(Rels::default());
        }
        Tree{ size: size, rels: r }
    }

    fn contains(&self, n: Node) -> bool {
        let (nx, ny) = n.data();
        nx >= 0 && ny >= 0
            && (nx as usize) < self.size
            && (ny as usize) < self.size
    }
    
    fn neighbors_dir(&self, n: Node) -> impl Iterator<Item = (Node, Dir)> + '_{
        // One might expect 'neighbors' to return the nodes of the tree that
        // are related to the given node, but that is not the case. Rather,
        // it returns the nodes of the tree that are adjacent to the input node
        // and thus potentially are related to the input node. The direction
        // denotes which direction the node lies in from the input node.
        //
        // This is the primitive traversing function for the tree upon which
        // other methods are built.
        Dir::dirs()
            .map(move |d| (n+d, d))
            .filter(move |&(v, _)| self.contains(v))
    }

    fn neighbors_rel(&self, n: Node) -> impl Iterator<Item = (Node, Rel)> + '_ {
        self.neighbors_dir(n)
            .map(move |(v, d)| (v, self[n][d]))
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
}

impl std::ops::Index<Node> for Tree {
    type Output = Rels;

    fn index(&self, n: Node) -> &Self::Output {
        &self.rels[(n.x as usize)*self.size + (n.y as usize)]
    } 
}

impl std::ops::IndexMut<Node> for Tree {
    fn index_mut(&mut self, n: Node) -> &mut Self::Output {
        &mut self.rels[(n.x as usize)*self.size + (n.y as usize)]
    } 
}


impl Tree {
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

pub struct Mask<T> {
    // We put a mask of colors on top of all the nodes of the tree.
    size: usize,
    mask: Vec<T>,
}

type Coloring = Mask<Color>;
type Tilings = Mask<Vec<usize>>;
type Partitions = Mask<Vec<Vec<usize>>>;

impl<T> Mask<T> where T: Default {
    fn new_mask(tree: &Tree) -> Self {
        let mut c = Vec::with_capacity(tree.size*tree.size);
        for _ in 0..tree.size*tree.size {
            c.push(T::default());
        }
        Mask{ size: tree.size, mask: c }
    }
}

impl<T> Mask<T> {
    pub fn new(size: usize, mask: Vec<T>) -> Self {
        assert_eq!(size*size, mask.len());
        Mask{ size, mask }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.mask.iter()
    }

    pub fn data(self) -> Vec<T> {
        self.mask
    }
}

impl<T: Eq + std::hash::Hash + Clone> Into<HashMap<T, Vec<Node>>> for Mask<T> {
    fn into(self) -> HashMap<T, Vec<Node>> {
        let mut map = HashMap::new();
        for x in 0..self.size {
            for y in 0..self.size {
                let n = Node::new(x as i32, y as i32);
                let nodes = map.entry(self[n].clone()).or_insert(Vec::new());
                nodes.push(n);
            }
        }
        map
    }
}

trait Reset {
    fn reset(&mut self);
}

impl<T> Reset for T where T: Default {
    fn reset(&mut self) {
        *self = T::default();
    }
}

impl<T> std::ops::Index<Node> for Mask<T> {
    type Output = T;
    fn index(&self, n: Node) -> &Self::Output {
        &self.mask[self.size*(n.x as usize) + (n.y as usize)]
    }
}

impl<T> std::ops::IndexMut<Node> for Mask<T> {
    fn index_mut(&mut self, n: Node) -> &mut Self::Output {
        &mut self.mask[self.size*(n.x as usize) + (n.y as usize)]
    }
}

impl Coloring {
    fn free_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        self.mask.iter()
            .enumerate()
            .filter(|&(_, &c)| c == Color::Free)
            .map(move |(i, _)| Node::new((i/self.size) as i32, (i%self.size) as i32) )
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
                    self.next = self.tree.children(current).filter(|&v| !self.visited[v]).next();
                    
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
    fn iter(&self) -> TreeIter {
        TreeIter{ next: Some(self.root()), visited: <Mask<bool>>::new_mask(self), tree: self }
    }
}

fn randomize_walk_from_tile_to_tree<R: rand::Rng> (
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
                tree.neighbors_dir(node).into_iter()
                    .filter(|&(v, _)|
                            coloring[v] != Color::Touched
                            && tree.neighbors_rel(v)
                            .filter(|&(_, r)| r == Rel::Child || r == Rel::Parent)
                            .count() < 3),
                rng
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
                },
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

fn fill_board_by_tree<R: rand::Rng> (size: usize, rng: &mut R)
                      -> (Tree, Node) {
    println!("starting 'fill_board_by_tree'...");
    // Returns the relationship graph of a tree filling a board of given size,
    // together with the root of said tree.
    let mut tree = Tree::new(size);
    let root = Node::new(rng.gen_range(0, size as i32), rng.gen_range(0, size as i32));
    // 'coloring' keeps track of which tiles are occupied, and to which capacity. 
    let mut occupancy = Coloring::new_mask(&tree);
    occupancy[root] = Color::Frozen;
 
    while occupancy.free_nodes().count() > 0 {
        // Seeing as there is at least one free node left, we can randomly pick one.
        let next_leaf: Node = *occupancy.free_nodes()
            .collect::<Vec<Node>>()
            .choose(rng)
            .unwrap();
        // Fill in random branch from leaf to the tree.
        randomize_walk_from_tile_to_tree(
            &mut tree,
            &mut occupancy,
            next_leaf,
            rng
        );
    }
    (tree, root)
}

fn _partitions(n: usize, parts: usize) -> Vec<Vec<usize>> {
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
    
    for i in 0..n+1 {
        for mut smaller_partition in _partitions(i, parts-1).into_iter() {
            smaller_partition.push(n-i);
            partitions.push(smaller_partition);
        }
    }
    partitions
}

// Returns the number of different tiles.
fn subdivide_tree<R: rand::Rng>(root: Node, tree: &Tree, rng: &mut R) -> Mask::<usize> {
    println!("starting 'subdivide_tree'...");
    let mut coloring = Mask::<usize>::new_mask(&tree);
    // Keeps track of the number of possible offspring in the same tile
    // as this node. The root of each tile contains all the possible
    // options for the size of the tile.
    let mut tilings = Tilings::new_mask(&tree);
    // Keeps track of which children should carry the part of the number
    // of tiles assigned to this node.
    let mut partitions = Partitions::new_mask(&tree);
    // Keeps track of whether the children of this node are satisfied/done.
    // A leaf is satisfied when the number of nodes assigned by its parent
    // are zero.
    let mut satisfied = Mask::<bool>::new_mask(&tree);

    tilings[root] = vec![2, 3];
    tilings[root].shuffle(rng);
    coloring[root] = 1;
    
    let mut next = Some(root);
    let mut tiles = 1;

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

        if let Some(partition) = partitions[current].last() {

            // We want to set all the children which are blank, which is indicated by
            // its coloring being 0. If the partition indicates that the current node
            // doesn't delegate the coloring of any nodes to a child we generate how
            // many that child is responsible for painting.
            for (&k, v) in partition.iter().zip(tree.children(current)){
                if coloring[v] == 0 {
                    if k > 0 {
                        tilings[v] = vec![k];
                        coloring[v] = coloring[current];
                    } else {
                        tilings[v] = vec![2,3];
                        tilings[v].shuffle(rng);
                        tiles += 1;
                        coloring[v] = tiles;
                    }
                }
            }

            // Next we want to mark this node as satisfied if all of its children are satisfied.
            // Note that a leaf that reaches this part of the code always has a valid partition
            // which means that it is not tasked with any tiling to resolve, so it is satisfied.
            if tree.children(current).all(|v| satisfied[v]) {
                satisfied[current] = true;
                next = tree.parent(current);
            } else {
                // 'next' has to be 'Some(_)' because there is some child 'v' such that
                // 'satisfied[v]' is false.
                next = tree.children(current).filter(|&v| !satisfied[v]).next();
            }
            
        } else if let Some(tiles) = tilings[current].pop() {
            // Having exhausted all the possible partitions of the current tilesize we
            // will have to try a new tilesize and partitions of said tilesize.
            partitions[current] = _partitions(tiles-1, tree.children(current).count());
            partitions[current].shuffle(rng);
        } else {
            // We have exhausted all possible number of tiles and distributions of them. 
            let parent = tree.parent(current).unwrap();
            partitions[parent].pop();
            reset_local_tree(parent, tree, &mut satisfied, &mut coloring, &mut tilings, &mut partitions);
            next = Some(parent);
        }
    }

    // We no longer need a color of 0 to signify that a tile is in initial state. Therefore we
    // shift down the color of every node by 1.
    for c in coloring.mask.iter_mut() {
        *c -= 1;
    }
    coloring
}

fn reset_local_tree(
    root: Node,
    tree: &Tree,
    satisfied: &mut Mask<bool>,
    coloring: &mut Mask<usize>,
    tilings: &mut Tilings,
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
                if coloring[v] != 0 {
                    satisfied[v].reset();
                    coloring[v].reset();
                    tilings[v].clear();
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
pub struct Boundary {
    // The nodes are stored in counter-clockwise order. The 'length' allows for
    // indexing the boundary by any integer (not necessarily positive) by having
    // the index repeat every 'length'.
    length: usize,
    nodes: Vec<Node>,
}

impl std::ops::Index<usize> for Boundary {
    type Output = Node;
    fn index(&self, i: usize) -> &Node {
        &self.nodes[i%self.length]
    }
}

impl std::ops::IndexMut<usize> for Boundary {
    fn index_mut(&mut self, i: usize) -> &mut Node {
        &mut self.nodes[i%self.length]
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
        let (fst_start, _) = self.nodes.iter().enumerate().cycle()
            .skip_while(|(_, v)| !snd.contains(v))
            .skip_while(|(_, v)| snd.contains(v))
            .next().unwrap();
        let fst_len = self.nodes.iter().cycle()
            .skip(fst_start)
            .take_while(|v| !snd.contains(v))
            .count() + 1;
        let fst_part: Vec<Node> = self.nodes.iter().cloned().cycle()
            .skip(fst_start)
            .take(fst_len)
            .collect();

        let (snd_start, _) = snd.nodes.iter().enumerate().cycle()
            .skip_while(|(_, v)| !self.contains(v))
            .skip_while(|(_, v)| self.contains(v))
            .next().unwrap();
        let snd_len = snd.nodes.iter().cycle()
            .skip(snd_start)
            .take_while(|v| !self.contains(v))
            .count() + 1;
        let snd_part: Vec<Node> = snd.nodes.iter().cloned().cycle()
            .skip(snd_start)
            .take(snd_len)
            .collect();
        
        let length = fst_part.len() + snd_part.len();
        let mut fusion = Vec::with_capacity(self.length + snd.length);
        fusion.extend(fst_part);
        fusion.extend(snd_part);
        
        // We update 'self' to be the fused boundary.
        *self = Self{ length: length, nodes: fusion };
    }

    fn from_node(n: Node) -> Self {
        let mut corners = Vec::with_capacity(4);
        corners.push(n);
        corners.push(n + Dir::S);
        corners.push(n + Dir::S + Dir::E);
        corners.push(n + Dir::E);
        Self{ length: 4, nodes: corners }
    }

    // We assume that 'coloring' is a mask on top of 'tree'.
    fn boundaries(tree: &Tree, coloring: &Mask<usize>) -> HashMap<usize, Self> {
        let mut boundaries: HashMap<usize, Self> = HashMap::new();

        for v in tree.iter() {
            if let Some(boundary) = boundaries.get_mut(&coloring[v]) {
                (*boundary).fuse(Self::from_node(v));
            } else {
                boundaries.insert(coloring[v], Self::from_node(v));
            }
        }

        boundaries
    }

    pub fn data(self) -> Vec<Node> {
        self.nodes
    }
}

pub fn generate<R: rand::Rng>(size: usize, rng: &mut R) -> (Mask<usize>, HashMap<usize, Boundary>) {
    let (tree, root) = fill_board_by_tree(size, rng);
    let coloring = subdivide_tree(root, &tree, rng);
    let boundaries = Boundary::boundaries(&tree, &coloring);
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
        DisplaySquare{chars}
    }
}

impl std::ops::Index<Node> for DisplaySquare {
    type Output = char;
    // The only valid indices are -1 <= x, y <= 1.
    fn index(&self, n: Node) -> &Self::Output {
        &self.chars[(3*(n.x + 1) + n.y + 1) as usize]
    }
}

impl std::ops::IndexMut<Node> for DisplaySquare {
    fn index_mut(&mut self, n: Node) -> &mut Self::Output {
        &mut self.chars[(3*(n.x + 1) + n.y + 1) as usize]
    }
}

impl DisplaySquare {
    fn line(&self, l: usize) -> String {
        // Crashes if l >= 3.
        let load = self.chars.chunks(3)
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
        }
        Rel::Parent => match d {
            Dir::N => 'v',
            Dir::S => std::char::from_u32(0x2227).unwrap(), // wedge
            Dir::W => '>',
            Dir::E => '<',
        }
    }
}

fn format_node(tree: &Tree, n: Node, c: char) -> DisplaySquare {
    let o = Node::default();
    let mut sq = DisplaySquare::default();
    sq[o] = c;
    for (d, r) in Dir::dirs().map(|d| (d, tree[n][d])) {
        sq[o+d] = format_rel(d, r);
    }
    sq
}

fn repr_tree(tree: &Tree) -> String {
    let mut repr = String::new();
    for y in 0..tree.size {
        // 'block' is the string corresponding to a row of nodes along the x axis
        let mut block = String::new();
        for l in 0..3 {
            for x in 0..tree.size {
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
    for y in 0..tree.size {
        // 'block' is the string corresponding to a row of nodes along the x axis
        let mut block = String::new();
        for l in 0..3 {
            for x in 0..tree.size {
                let n = Node::new(x as i32, y as i32);
                block += &format_node(tree, n, std::char::from_digit((tiling[n]%16) as u32, 16).unwrap()).line(l);
                block += " ";
            }
            block += "\n";
        }
        repr += block.as_ref();
        repr += "\n";
    }
    repr
}
