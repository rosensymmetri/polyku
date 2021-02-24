use crate::mask::*;
use abstalg::{Domain, Polynomials, UnitaryRing};
use itertools::Itertools;
use rand::prelude::*;
use std::collections::HashMap;

fn place_coefficients<R: rand::Rng, S: abstalg::UnitaryRing>(
    size: usize,
    coefficients: &Vec<S::Elem>,
    rng: &mut R,
) -> Mask<Poly<S>> {
    // We expect that there are two coefficients for every tile.
    assert_eq!(2 * size, coefficients.len());

    let assignments = place_numbers(size, rng);

    let polys = assignments
        .chunks(2)
        .map(|chunk| {
            vec![
                coefficients[chunk[0]].clone(),
                coefficients[chunk[1]].clone(),
            ]
        })
        .collect();
    Mask::new(size, polys)
}

struct NumSubset {
    size: usize,
    elts: Vec<bool>,
}

impl NumSubset {
    fn full(size: usize) -> Self {
        let mut elts = Vec::with_capacity(size);
        for _ in 0..size {
            elts.push(true);
        }
        NumSubset { size, elts }
    }

    fn has(&self, idx: usize) -> bool {
        self.elts[idx]
    }

    fn rm(&mut self, idx: usize) {
        self.elts[idx] = false;
    }

    fn vec(&self) -> Vec<usize> {
        (0..self.size)
            .into_iter()
            .filter(|&i| self.elts[i])
            .collect()
    }
}

fn place_numbers<R: rand::Rng>(size: usize, rng: &mut R) -> Vec<usize> {
    let mut rows: Vec<usize>;

    loop {
        rows = Vec::new();
        let mut cols = Vec::new();
        for _ in 0..size {
            cols.push(NumSubset::full(2 * size));
        }

        // Each row is processed one at a time.
        for _ in 0..size {
            let mut candidates = NumSubset::full(2 * size);
            let mut indices = NumSubset::full(2 * size);
            let mut row = Vec::new();
            for _ in 0..2 * size {
                row.push(0);
            }

            let order = {
                let mut o: Vec<usize> = (0..2 * size).into_iter().collect();
                o.shuffle(rng);
                o
            };

            // first one in candidates which is free and not used above it
            for _ in 0..2 * size {
                let maybe = order
                    .iter()
                    .filter(|&&i| candidates.has(i))
                    .cartesian_product(indices.vec())
                    .filter(|&(&i, x)| cols[x / 2].has(i))
                    .filter(|&(&_, x)| {
                        (0..2 * size)
                            .into_iter()
                            .filter(|&i| candidates.has(i) && cols[x / 2].has(i))
                            .count()
                            == 1
                    })
                    .chain(
                        order
                            .iter()
                            .filter(|&&i| candidates.has(i))
                            .cartesian_product(indices.vec())
                            .filter(|&(&i, x)| cols[x / 2].has(i))
                            .filter(|&(&i, _)| {
                                (0..size)
                                    .into_iter()
                                    .filter(|&col| {
                                        (indices.has(2 * col) || indices.has(2 * col + 1))
                                            && cols[col].has(i)
                                    })
                                    .count()
                                    == 1
                            }),
                    )
                    .chain(
                        order
                            .iter()
                            .filter(|&&i| candidates.has(i))
                            .cartesian_product(indices.vec())
                            .filter(|&(&i, x)| cols[x / 2].has(i))
                            .min_by_key(|&(&i, x)| {
                                (0..2 * size)
                                    .into_iter()
                                    .filter(|&i| candidates.has(i) && cols[x / 2].has(i))
                                    .count()
                                    + (0..size)
                                        .into_iter()
                                        .filter(|&col| {
                                            (indices.has(2 * col) || indices.has(2 * col + 1))
                                                && cols[col].has(i)
                                        })
                                        .count()
                            })
                            .into_iter(),
                    )
                    .next();
                if let Some((&i, x)) = maybe {
                    cols[x / 2].rm(i);
                    candidates.rm(i);
                    indices.rm(x);
                    row[x] = i;
                } else {
                    continue;
                }
            }
            rows.extend(row);
        }
        break;
    }
    rows
}

#[derive(Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
}

impl Op {
    fn new(i: usize) -> Self {
        match i % 3 {
            0 => Self::Add,
            1 => Self::Sub,
            2 => Self::Mul,
            _ => unreachable!(),
        }
    }
}

type Poly<S> = Vec<<S as Domain>::Elem>;

fn apply_op_to_polynomials<S>(
    op: Op,
    ring: &Polynomials<S>,
    polys: &Vec<Poly<S>>,
) -> Option<Poly<S>>
where
    S: UnitaryRing,
{
    match op {
        Op::Add => Some(polys.into_iter().fold(Polynomials::zero(ring), |val, p| {
            Polynomials::add(ring, &val, p)
        })),
        Op::Mul => Some(polys.into_iter().fold(Polynomials::one(ring), |val, p| {
            Polynomials::mul(ring, &val, p)
        })),
        Op::Sub => {
            if let [ref p0, ref p1] = polys[..] {
                Some(Polynomials::sub(ring, &p0, &p1))
            } else {
                None
            }
        }
    }
}

fn group_coefficients_by_tile<S, R>(
    size: usize,
    tiles: &HashMap<usize, Vec<Node>>,
    ring: &Polynomials<S>,
    polys: &Mask<Poly<S>>,
    rng: &mut R,
) -> HashMap<usize, (Op, Poly<S>)>
where
    S: UnitaryRing,
    R: rand::Rng,
{
    let mut grouped = HashMap::new();

    for (&k, nodes) in tiles.iter() {
        let current_polys = &nodes.into_iter().map(|&v| polys[v].clone()).collect();
        let mut op: Op;
        let res: Poly<S>;
        loop {
            op = Op::new(rng.gen());
            if let Some(poly) = apply_op_to_polynomials(op, ring, current_polys) {
                res = poly;
                break;
            }
        }

        grouped.insert(k, (op, res));
    }

    grouped
}

pub(crate) fn generate<S: UnitaryRing, R: rand::Rng>(
    size: usize,
    tiles: &HashMap<usize, Vec<Node>>,
    base_ring: S,
    coefficients: &Vec<S::Elem>,
    rng: &mut R,
) -> (Mask<Poly<S>>, HashMap<usize, (Op, Poly<S>)>) {
    let ring = Polynomials::new(base_ring);
    place_numbers(size, rng);
    let answers: Mask<Poly<S>> = place_coefficients::<R, S>(size, coefficients, rng);
    let clues = group_coefficients_by_tile(size, tiles, &ring, &answers, rng);
    (answers, clues)
}
