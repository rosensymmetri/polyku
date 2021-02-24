use rand::seq::SliceRandom;
use std::ops::{Index, IndexMut};
use std::hash::{Hash, Hasher};
use std::collections::{HashMap, HashSet};

#[derive(PartialEq,Copy,Clone)]
enum Edge {
    Free,
    Closed,
    Locked,
    Selected,
}

#[derive(PartialEq,Copy,Clone)]
enum Dir {
    N,
    S,
    W,
    E,
}

impl Dir {
    // fn cases() -> [Dir; 4] {
    //    [Dir::N, Dir::S, Dir::W, Dir::E]
    // }
    
    fn spin_pos(self) -> Self {
        match self {
            Dir::N => Dir::W,
            Dir::S => Dir::E,
            Dir::W => Dir::S,
            Dir::E => Dir::N,
        }
    }

    fn spin_neg(self) -> Self {
        match self {
            Dir::N => Dir::E,
            Dir::S => Dir::W,
            Dir::W => Dir::N,
            Dir::E => Dir::S,
        }
    }

    fn spin_opp(self) -> Self {
        match self {
            Dir::N => Dir::S,
            Dir::S => Dir::N,
            Dir::W => Dir::E,
            Dir::E => Dir::W,
        }
    }
}

#[derive(Copy,Clone)]
struct EdgeIndex {
    i: usize,
    j: usize,
    dir: Dir,
}

impl EdgeIndex {
    fn head(self) -> (usize, usize) {
        match self.dir {
            Dir::N => (self.i, self.j+1),
            Dir::S => (self.i, self.j-1),
            Dir::W => (self.i-1, self.j),
            Dir::E => (self.i+1, self.j)
        }
    }

    fn tail(self) -> (usize, usize) {
        (self.i, self.j)
    }

    fn ahead(self) -> Self {
        let (i, j) = self.head();
        EdgeIndex{i, j, dir: self.dir}
    }

    fn left(self) -> Self {
        let (i, j) = self.head();
        EdgeIndex{i, j, dir: self.dir.spin_pos()}
    }
    
    fn right(self) -> Self {
        let (i, j) = self.head();
        EdgeIndex{i, j, dir: self.dir.spin_neg()}
    }

    fn tail_left(self) -> Self {
        let (i, j) = self.tail();
        EdgeIndex{i, j, dir: self.dir.spin_pos()}
    }

    fn tail_back(self) -> Self {
        let (i, j) = self.tail();
        EdgeIndex{i, j, dir: self.dir.spin_opp()}
    }
}

impl PartialEq for EdgeIndex {
    fn eq(&self, other: &Self) -> bool {
        (self.head() == other.head() && self.tail() == other.tail())
            || (self.head() == other.tail() && self.tail() == other.head())
    }
}

impl Eq for EdgeIndex {}

impl Hash for EdgeIndex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (i1, j1) = self.head();
        let (i2, j2) = self.tail();

        let min_x = std::cmp::min(i1, i2);
        let max_x = std::cmp::max(i1, i2);

        let min_y = std::cmp::min(j1, j2);
        let max_y = std::cmp::max(j1, j2);

        (min_x, min_y).hash(state);
        (max_x, max_y).hash(state);
    }
}


struct EdgeColoring {
    edges: Vec<Edge>,
    n: usize,
}

impl EdgeColoring {
    fn pure_index(&self, index: EdgeIndex) -> usize {
        let n = self.n;
        
        match index.dir {
            Dir::N => (index.j-1)*(2*n+1) + n + index.i,
            Dir::S => index.j*(2*n+1)     + n + index.i,
            Dir::W => index.j*(2*n+1)     + index.i-1,
            Dir::E => index.j*(2*n+1)     + index.i,
        }
    }

    fn edges_from_coordinate(&self, i: usize, j: usize) -> [Edge; 4] {
        [
            self[EdgeIndex{i,j, dir: Dir::N}],
            self[EdgeIndex{i,j, dir: Dir::S}],
            self[EdgeIndex{i,j, dir: Dir::E}],
            self[EdgeIndex{i,j, dir: Dir::W}],
        ]
    }

    fn edges_from_edge(&self, index: EdgeIndex) -> Vec<(EdgeIndex, Edge)> {
        vec![
            (index.ahead(), self[index.ahead()]),
            (index.ahead(), self[index.left()]),
            (index.ahead(), self[index.right()]),
        ]
    }
}

impl Index<EdgeIndex> for EdgeColoring {
    type Output = Edge;
    
    fn index(&self, index: EdgeIndex) -> &Self::Output {
        &self.edges[self.pure_index(index)]
    }
}

impl IndexMut<EdgeIndex> for EdgeColoring {
    fn index_mut(&mut self, index: EdgeIndex) -> &mut Self::Output {
        let index = self.pure_index(index);
        &mut self.edges[index]
    }

}

impl EdgeColoring {
    fn new_blank(n: usize) -> Self {
        if n == 0 {
            panic!()
        }

        let mut edges = Vec::with_capacity( n*(2*n+1) + (n+1) );

        for _row in 0..n {
            for _horizontal_column in 0..n {
                edges.push(Edge::Closed);
            }
            edges.push(Edge::Closed);
            for _inner_vertical_column in 1..n {
                edges.push(Edge::Free);
            }
            edges.push(Edge::Closed);
        }
        for _horizontal_column in 0..n {
            edges.push(Edge::Closed);
        }

        
        Self {
            edges,
            n,
        }
    }

}

// 
struct Tiling {
    squares: Vec<usize>,
    tiles: usize,
    coloring: EdgeColoring,
    outer_boundary: HashMap<usize, Vec<EdgeIndex>>,
    inner_boundary: HashMap<usize, Vec<EdgeIndex>>,
}

struct Tile {
}

impl Tiling {
    fn new_blank(n: usize) -> Self {
        let mut squares = Vec::with_capacity(n*n);

        for _ in 0..n*n {
            squares.push(0);
        }

        let mut inner_boundary = HashSet::with_capacity( n*(2*n+1) + (n+1) - 4*n );
        for inner in 1..n+1 {
            inner_boundary.insert(EdgeIndex{ i: inner, j: 0, dir: Dir::S });
        }
        for inner in 1..n+1 {
            inner_boundary.insert(EdgeIndex{ i: n+1, j: inner, dir: Dir::W });
        }
        for inner in 1..n+1 {
            inner_boundary.insert(EdgeIndex{ i: inner, j: n+1, dir: Dir::N });
        }
        for inner in 1..n+1 {
            inner_boundary.insert(EdgeIndex{ i: 0, j: inner, dir: Dir::E });
        }
        
        Tiling {
            squares,
            tiles: 1,
            coloring: EdgeColoring::new_blank(n),
        }
    }

    // We assume that the outer boundary is a closed loop and the edges are
    // given in a positively oriented order.
    fn inner_boundary(&self, tile: usize) -> Vec<EdgeIndex> {
        // Take each edge and make the free edges next to that one, that are to the
        // 'left' of the edge, into the inner boundary edges.
        self.outer_boundary[&tile]
            .iter()
            .map(|&e| [e.tail_left(), e.tail_back()]
              .into_iter()
              .take_while(|&&t| self.coloring[t] == Edge::Free)   
            )
            .flatten()
            .map(|&e| e)
            .collect()
    }

    // Computes the number of squares in a tile. It performs this computation
    // by traversing the boundary of the square.
    fn tile_size(&self, tile: usize) -> usize {
        let pos = Dir::N;
        let neg = Dir::S;

        // The result will end up positive because of the positive orientation
        // of the boundary but it can briefly become negative during the
        // computation (because we may hit many south facing edges before hitting
        // north facing ones). 
        let squares: isize = 0;

        // We add the 'x' coordinate of the edges facing NS, using a sign to
        // account for the direction. We count the number of squares between each
        // pair of NS edges along the same row, where due to orientation, the N
        // edge is to the right of the S edge.
        for edge in self.outer_boundary[&tile] {
            match edge.dir {
                Dir::N => squares += edge.i as isize,
                Dir::S => squares -= edge.i as isize,
                _ => {},
            }
        }

        assert!(squares > 0);
        squares as usize
    }

    // Randomly split the given tile into two. Will scratch and retry until
    // the new tiles have proper sizes. Panics if 'tile' is outside permitted
    // range or if called on a tile with empty inner boundary.
    pub fn split(&mut self, tile: usize, rng: rand::rngs::StdRng) -> Option<()> {
        assert!(tile < self.tiles);

        let mut current_edge: EdgeIndex = *self.inner_boundary(tile)
            .choose(&mut rng)?;

        let split_path = self.splitting_path_from_edge(tile, current_edge, rng)?;
        let split_path_last = split_path.len()-1;
        
        let (start_index, _) = self.outer_boundary[&tile].iter()
            .enumerate()
            .find(|(_, e)| e.tail() == split_path[0].tail())?;
        self.outer_boundary[&tile].rotate_left(start_index);
        
        let (end_index, _) = self.outer_boundary[&tile].iter()
            .enumerate()
            .find(|(_, e)| e.tail() == split_path[split_path_last].tail())?;

        let boundary_of_old_tile = self.outer_boundary[&tile][0..end_index].to_vec();
        let boundary_of_new_tile = self.outer_boundary[&tile][end_index..].to_vec();
        
        self.tiles += 1;

        // Finally build the boundary of the new tile!
        self.outer_boundary[&self.tiles] = split_path.clone()
            .into_iter()
            .chain(
                boundary_of_new_tile.into_iter()
            ).collect();

        // And fix the boundary of the old tile. Note that to have the
        // correct orientation, we need to reverse the splitting path.
        self.outer_boundary[&tile] = split_path.into_iter()
            .rev()
            .chain(
                boundary_of_old_tile.into_iter()
            ).collect();
        

        for &mut edge in split_path.iter_mut() {
            
        }
        
        Some(())
    }
    
    pub fn splitting_path_from_edge(
        &mut self,
        tile: usize,
        current_edge: EdgeIndex,
        rng: rand::rngs::StdRng
    )
        -> Option<Vec<EdgeIndex>> {
        let mut traversed_path = Vec::<EdgeIndex>::new();
        
        loop {
            traversed_path.push(current_edge);
            self.edge_grid[current_edge] = Edge::Locked;
            
            if ! self.inner_boundary(tile).contains(&current_edge) {
                break;
            }

            // Select the next 'current_edge'.
            let (next_edge, _) = *self.edge_grid
                .edges_from_edge(current_edge)
                .into_iter()
                .filter(|(_,e)| *e == Edge::Free)
                .collect::<Vec<(EdgeIndex, Edge)>>()
                .choose(&mut rng)?;
            
            // Mark the edges that we have touched as Selected, so that we can avoid double crossings. 
            for (_, edge) in self.edge_grid.edges_from_edge(current_edge).iter_mut() {
                match edge {
                    Edge::Free => *edge = Edge::Selected,
                    _ => {},
                }
            }

            current_edge = next_edge;
        }

        // If we exit the loop without recieving an error we have that 'traversed_path' is a path
        // which begins and ends on the edge of some tile. We can now return the traversed path.
        Some(traversed_path)
    }
}
