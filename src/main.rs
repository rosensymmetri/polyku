use rand::prelude::*;
use std::collections::HashMap;

use abstalg::Integers;
use num::bigint::Sign;
use num::BigInt;

mod mask; // defines nodes and masks over squares of nodes
mod poly;
mod tex;
mod tree; // for generating the tiling // for generating the polynomials

fn generate_random_polyku_pdf(size: usize, coeffs: Vec<isize>) -> Vec<u8> {
    assert_eq!(2 * size, coeffs.len());
    let coeffs_bigint = coeffs.iter().map(|&i| BigInt::from(i)).collect();
    let base_ring = Integers();

    let mut rng = rand::thread_rng();
    let (tiles, boundaries) = tree::generate(size, &mut rng);

    let tiles: HashMap<usize, Vec<mask::Node>> = tiles.into();
    let (ans, clues) = poly::generate(size, &tiles, base_ring, &coeffs_bigint, &mut rng);

    let clues: HashMap<usize, (String, String)> = clues
        .into_iter()
        .map(|(key, (op, clue))| (key, (op_tex_string(op), to_string(&clue))))
        .collect();
    let ans: Vec<String> = ans
        .data()
        .into_iter()
        .map(|poly| to_string(&poly))
        .collect();

    tex::pdf_doc(size, coeffs, tiles, clues, boundaries, ans)
}

fn main() -> std::io::Result<()> {
    let size = 3;
    let coeffs = vec![1, 2, 3, 4, 5, 6];

    std::fs::write("test.pdf", generate_random_polyku_pdf(size, coeffs))?;

    Ok(())
}

fn head_to_string(int: &BigInt) -> Option<String> {
    let (sgn, num) = int.to_u32_digits();
    if num.len() == 0 {
        return None;
    }

    match num[0] {
        0 => None,
        _ => Some(((num[0] as isize) * sign(sgn)).to_string()),
    }
}

fn tail_to_string(int: &BigInt) -> Option<String> {
    let (sgn, num) = int.to_u32_digits();
    if num.len() == 0 {
        return None;
    }

    match num[0] {
        0 => None,
        1 => Some(" ".to_string() + sign_str(sgn) + " "),
        n => Some(" ".to_string() + sign_str(sgn) + " " + n.to_string().as_str()),
    }
}

fn exp(var: char, k: usize) -> String {
    match k {
        0 => "".to_string(),
        1 => var.to_string(),
        _ => format!("{}^{}", var, k),
    }
}

fn to_string(poly: &Vec<BigInt>) -> String {
    // a + b*x + c*x^2 + ...
    match poly[..] {
        [] => "0".to_string(),
        _ => poly
            .iter()
            .enumerate()
            .map(|(k, int)| {
                if k == 0 {
                    head_to_string(int)
                } else {
                    tail_to_string(int)
                }
                .map(|s| s + exp('x', k).as_str())
            })
            .flatten()
            .collect(),
    }
}

fn sign(s: num::bigint::Sign) -> isize {
    match s {
        num::bigint::Sign::Plus | num::bigint::Sign::NoSign => 1,
        num::bigint::Sign::Minus => -1,
    }
}

fn sign_str(s: num::bigint::Sign) -> &'static str {
    match s {
        num::bigint::Sign::Plus => "+",
        num::bigint::Sign::NoSign => "",
        num::bigint::Sign::Minus => "-",
    }
}

fn op_tex_string(op: poly::Op) -> String {
    match op {
        poly::Op::Add => "+".to_string(),
        poly::Op::Sub => "-".to_string(),
        poly::Op::Mul => r"\times".to_string(),
    }
}
