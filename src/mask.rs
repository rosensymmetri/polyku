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
pub struct Node {
    x: i32,
    y: i32,
}

impl Node {
    pub(crate) fn new(x: i32, y: i32) -> Self {
        Node{x,y}
    }

    pub fn data(self) -> (i32, i32) {
        (self.x, self.y)
    }
}

impl std::ops::Add<Dir> for Node {
    type Output = Node;
    fn add(self, d: Dir) -> Node {
        match d {
            Dir::N => Node::new(self.x, self.y-1),
            Dir::S => Node::new(self.x, self.y+1),
            Dir::W => Node::new(self.x-1, self.y),
            Dir::E => Node::new(self.x+1, self.y),
        }
    }
}
