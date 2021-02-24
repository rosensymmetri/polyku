use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dir {
    N,
    S,
    W,
    E,
}

impl Dir {
    pub(crate) fn num(self) -> usize {
        match self {
            Self::N => 0,
            Self::S => 1,
            Self::W => 2,
            Self::E => 3,
        }
    }

    pub(crate) fn opp(self) -> Self {
        match self {
            Self::N => Self::S,
            Self::S => Self::N,
            Self::W => Self::E,
            Self::E => Self::W,
        }
    }

    pub(crate) fn dirs() -> impl Iterator<Item = Dir> {
        vec![Self::N, Self::S, Self::W, Self::E].into_iter()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(crate) struct Node {
    x: i32,
    y: i32,
}

impl Node {
    pub(crate) fn new(x: i32, y: i32) -> Self {
        Node { x, y }
    }

    pub(crate) fn data(self) -> (i32, i32) {
        (self.x, self.y)
    }
}

impl std::ops::Add<Dir> for Node {
    type Output = Node;
    fn add(self, d: Dir) -> Node {
        match d {
            Dir::N => Node::new(self.x, self.y - 1),
            Dir::S => Node::new(self.x, self.y + 1),
            Dir::W => Node::new(self.x - 1, self.y),
            Dir::E => Node::new(self.x + 1, self.y),
        }
    }
}

pub(crate) struct Mask<T> {
    // We put a mask of colors on top of all the nodes of the tree.
    size: usize,
    mask: Vec<T>,
}

impl<T> Mask<T> {
    pub(crate) fn new_default(size: usize) -> Self
    where
        T: Default,
    {
        let mut mask = Vec::with_capacity(size * size);
        for _ in 0..size * size {
            mask.push(T::default());
        }
        Mask { size, mask }
    }

    pub(crate) fn new(size: usize, mask: Vec<T>) -> Self {
        assert_eq!(size * size, mask.len());
        Mask { size, mask }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.mask.iter()
    }

    pub(crate) fn iter_nodes(&self) -> impl Iterator<Item = (Node, &T)> + '_ {
        self.mask
            .iter()
            .enumerate()
            .map(move |(i, t)| (Node::new((i / self.size) as i32, (i % self.size) as i32), t))
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn data(self) -> Vec<T> {
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

impl<T: Eq + std::hash::Hash + Clone> Mask<T> {
    pub(crate) fn export(self) -> HashMap<T, Vec<(i32, i32)>> {
        let mut map = HashMap::new();
        for x in 0..self.size {
            for y in 0..self.size {
                let n = Node::new(x as i32, y as i32);
                let nodes = map.entry(self[n].clone()).or_insert(Vec::new());
                nodes.push((x as i32, y as i32));
            }
        }
        map
    }
}

trait Reset {
    fn reset(&mut self);
}

impl<T> Reset for T
where
    T: Default,
{
    fn reset(&mut self) {
        *self = T::default();
    }
}

impl<T> std::ops::Index<Node> for Mask<T> {
    type Output = T;
    fn index(&self, n: Node) -> &Self::Output {
        &self.mask[self.size * (n.x as usize) + (n.y as usize)]
    }
}

impl<T> std::ops::IndexMut<Node> for Mask<T> {
    fn index_mut(&mut self, n: Node) -> &mut Self::Output {
        &mut self.mask[self.size * (n.x as usize) + (n.y as usize)]
    }
}
