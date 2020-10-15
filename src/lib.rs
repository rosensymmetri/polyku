
use rand::prelude::*;
use std::collections::HashMap;

use num::BigInt;
use num::bigint::Sign;
use abstalg::Integers;

use pyo3::prelude::*;

mod mask; // defines nodes and masks over squares of nodes
mod tree; // for generating the tiling
mod poly; // for generating the polynomials


#[pymodule]
fn polyku_gen(py: Python, m: &PyModule) -> PyResult<()> {
    
    #[pyfn(m, "generate_random_polyku")]
    fn generate_random_polyku_py(_py: Python, size: usize, coeffs: Vec<isize>) -> 
    (HashMap<usize, Vec<(usize, usize)>>,
    HashMap<usize, (String, String)>,
    HashMap<usize, Vec<(usize, usize)>>,
    Vec<String>,) {
        assert_eq!(2*size, coeffs.len());
        let coeffs = &coeffs.into_iter().map(|i| BigInt::from(i)).collect();
        let base_ring = Integers();

        let mut rng = rand::thread_rng();
        
        let (coloring, mut boundaries) = tree::generate(size, &mut rng);
        let mut tiles: HashMap<usize, Vec<tree::Node>> = coloring.into();
        let (mut ans, mut clues) = poly::generate(size, &tiles, base_ring, coeffs, &mut rng);


        // We 'flatten' the data before sending it to Python.
        let tiles: HashMap<usize, Vec<(usize, usize)>> = tiles.drain()
            .map(|(key, vec)| (key,
                               vec.into_iter()
                               .map(|v| {
                                   let (x, y) = v.data();
                                   (x as usize, y as usize) } )
                               .collect() )
            )
            .collect();
        let clues: HashMap<usize, (String, String)> = clues.drain()
            .map(|(key, (op, clue))| (key, (op_string(op), to_string(&clue))) )
            .collect();
        let boundaries: HashMap<usize, Vec<(usize, usize)>> = boundaries.drain()
            .map(|(key, bdy)| (key,
                               bdy.data().into_iter()
                               .map(|v| {
                                   let (x, y) = v.data();
                                   (x as usize, y as usize) } )
                               .collect() )
            )
            .collect();
        let ans: Vec<String> = ans.data().into_iter().map(|poly| to_string(&poly)).collect();
        
        (tiles, clues, boundaries, ans)
    }

    Ok(())
}

fn main() {
    let size: usize = 6;
    let coeffs = vec![1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6,];
    let coeffs = &coeffs.into_iter().map(|i| BigInt::from(i)).collect();
    let base_ring = Integers();
    
    let mut rng = thread_rng();
    let (coloring, _boundaries) = tree::generate(size, &mut rng);
    println!("main: board has been split into tiles");
    let tiling: HashMap<usize, Vec<tree::Node>> = coloring.into();
    let (ans, _clues) = poly::generate(size, &tiling, base_ring, coeffs, &mut rng);

    let display_ans: Vec<String> = ans.iter().map(|poly| to_string(poly)).collect();
    println!("    [");
    for y in 0..size {
        println!("     ; {:?}", &display_ans[y*size .. (y+1)*size]);
    }
    println!("    ]");
}


fn head_to_string(int: &BigInt) -> Option<String> {
    let (sgn, num) = int.to_u32_digits();
    if num.len() == 0 { return None; }
    
    match num[0] {
        0 => None,
        _ => Some(((num[0] as isize) * sign(sgn)).to_string())
    }
}

fn tail_to_string(int: &BigInt) -> Option<String> {
    let (sgn, num) = int.to_u32_digits();
    if num.len() == 0 { return None; }
    
    match num[0] {
        0 => None,
        1 => Some(" ".to_string() + sign_str(sgn) + " "),
        n => Some(" ".to_string() + sign_str(sgn) + " " + n.to_string().as_str()),
    }
}

fn exp(c: char, k: usize) -> String {
    match k {
        0 => "".to_string(),
        1 => c.to_string(),
        _ => c.to_string() + "^" + k.to_string().as_str(),
    }
}

fn to_string(poly: &Vec<BigInt>) -> String {
    // a + b*x + c*x^2 + ...
    match poly[..] {
        [] => "0".to_string(),
        _ => {
            poly.iter().enumerate()
                .map(|(k, int)|
                     if k == 0 { head_to_string(int) } else { tail_to_string(int) }
                     .map(|s| s + exp('x', k).as_str())
                ).flatten()
                .collect()
        }
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


fn op_string(op: poly::Op) -> String {
    match op {
        poly::Op::Add => "+".to_string(),
        poly::Op::Sub => "-".to_string(),
        poly::Op::Mul => "*".to_string(),
    }
}
