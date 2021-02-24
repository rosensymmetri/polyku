use crate::mask::Node;
use std::collections::HashMap;
use texrender::{elems, tpl, tpl::elements::*};

type TexObj = Box<dyn tpl::TexElement>;

struct TexCmd {
    ident: String,
    stored_opts: Vec<TexObj>,
    stored_args: Vec<TexObj>,
    inline: bool,
}

impl TexCmd {
    fn arg<T: tpl::IntoTexElement>(mut self, elem: T) -> Self {
        self.stored_args.push(elem.into_tex_element());
        self
    }

    fn opt<T: tpl::IntoTexElement>(mut self, elem: T) -> Self {
        self.stored_opts.push(elem.into_tex_element());
        self
    }
}

fn texcmd(s: &str) -> TexCmd {
    TexCmd {
        ident: s.to_string(),
        stored_opts: Vec::new(),
        stored_args: Vec::new(),
        inline: false,
    }
}

fn texinl(s: &str) -> TexCmd {
    TexCmd {
        ident: s.to_string(),
        stored_opts: Vec::new(),
        stored_args: Vec::new(),
        inline: true,
    }
}

impl tpl::IntoTexElement for TexCmd {
    fn into_tex_element(self) -> TexObj {
        tpl::MacroCall::new(
            self.ident,
            tpl::OptArgs::new(self.stored_opts),
            tpl::Args::new(self.stored_args),
        )
        .into_tex_element()
    }
}

struct TexEnv {
    ident: String,
    stored_opts: Vec<TexObj>,
    stored_args: Vec<TexObj>,
    children: Vec<TexObj>,
}

impl TexEnv {
    fn arg<T: tpl::IntoTexElement>(mut self, elem: T) -> Self {
        self.stored_args.push(elem.into_tex_element());
        self
    }
}

fn texenv(s: &str, children: Vec<TexObj>) -> TexEnv {
    TexEnv {
        ident: s.to_string(),
        stored_opts: Vec::new(),
        stored_args: Vec::new(),
        children,
    }
}

impl tpl::IntoTexElement for TexEnv {
    fn into_tex_element(self) -> TexObj {
        tpl::BeginEndBlock::new(
            self.ident,
            tpl::OptArgs::new(self.stored_opts),
            tpl::Args::new(self.stored_args),
            self.children,
        )
        .into_tex_element()
    }
}

static TILE_WIDTH: f32 = 3.0;

fn tikz_at(n: Node, size: usize) -> String {
    let (x, y) = n.data();

    let (x, y) = (x as f32, y as f32);
    let size = size as f32;
    /*"(".to_string()
    + (TILE_WIDTH * x as f32).to_string().as_ref()
    + ", "
    + (TILE_WIDTH * (size as f32 - y as f32)).to_string().as_ref()
    + ")"*/
    format!("({} ,{})", TILE_WIDTH * x, TILE_WIDTH * (size - y))
}

fn smallest_node(node_list: Vec<Node>) -> Node {
    let (mut minx, mut miny) = (i32::MAX, i32::MAX);

    for n in node_list {
        let (x, y) = n.data();
        if y < miny || (y == miny && x < minx) {
            minx = x;
            miny = y;
        }
    }
    Node::new(minx, miny)
}

fn centering(children: Vec<Box<dyn tpl::TexElement>>) -> tpl::BeginEndBlock {
    tpl::BeginEndBlock::new(
        "centering",
        tpl::OptArgs::default(),
        tpl::Args::default(),
        children,
    )
}

fn block(children: Vec<Box<dyn tpl::TexElement>>) -> tpl::AnonymousBlock {
    tpl::AnonymousBlock::new(children)
}

fn tikz_path(p: Vec<Node>, size: usize) -> String {
    let head = p[0];
    let mut path: String = p
        .into_iter()
        .map(|n| tikz_at(n, size))
        .map(|mut s| {
            s.push_str(" -- ");
            s
        })
        .collect();

    path.push_str(&tikz_at(head, size));

    path
}

fn tex_array(coeffs: Vec<isize>) -> String {
    let n = coeffs.len() - 1;
    let last = coeffs[n];

    let mut list = "{[}".to_string();
    for c in coeffs.into_iter().take(n) {
        list.push_str(&format!("{}, ", c));
    }

    list.push_str(&format!("{}", last));
    list.push_str("{]}");

    list
}

pub(crate) fn tex_doc(
    size: usize,
    coeffs: Vec<isize>,
    tiles: HashMap<usize, Vec<Node>>,
    clues: HashMap<usize, (String, String)>,
    boundaries: HashMap<usize, Vec<Node>>,
    ans: Vec<String>,
) -> String {
    let tikz_gridx: Vec<TexObj> = (0..=size)
        .into_iter()
        .map(|x| x as i32)
        .map(|x| {
            elems!(
                texcmd("draw").opt("semithick"),
                tikz_path(vec![Node::new(x, 0), Node::new(x, size as i32)], size),
                ";\n"
            )
        })
        .flatten()
        .collect();

    let tikz_gridy: Vec<TexObj> = (0..=size)
        .into_iter()
        .map(|y| y as i32)
        .map(|y| {
            elems!(
                texcmd("draw").opt("semithick"),
                tikz_path(vec![Node::new(0, y), Node::new(size as i32, y)], size),
                ";\n"
            )
        })
        .flatten()
        .collect();

    let tikz_boundaries: Vec<Box<dyn tpl::TexElement>> = boundaries
        .into_iter()
        .map(|(_, bs)| {
            elems!(
                texinl("draw").opt("ultra thick"),
                tikz_path(bs, size),
                ";\n"
            )
            .into_iter()
        })
        .flatten()
        .collect();

    // The hashmap access does not panic because we assume that the maps passed
    // as arguments have values at the same entries.
    let clues: Vec<(Node, String, String)> = tiles
        .into_iter()
        .map(|(k, v)| {
            let (op, clue) = clues[&k].clone();
            let n = smallest_node(v);
            (n, op, clue)
        })
        .collect();

    let tikz_clues: Vec<TexObj> = clues
        .into_iter()
        .map(|(n, op, clue)| {
            elems!(
                texinl("node")
                    .opt("anchor=north west")
                    .opt(format!("text width={}cm", TILE_WIDTH)),
                format!(" at {}", tikz_at(n, size)),
                block(elems!(texenv(
                    "tabular",
                    elems!(raw(format!(r"${}$ \\ ${}$", clue, op)))
                )
                .arg(raw(format!(
                    r">{{\raggedright}}p{{{}cm}}",
                    TILE_WIDTH as f32 - 0.5
                ))))),
                raw(";")
            )
        })
        .flatten()
        .collect();

    let tex = doc(elems!(
        documentclass(elems!(), "article"),
        texcmd("usepackage").opt("T1").arg("fontenc"),
        texcmd("usepackage").opt("utf8").arg("inputenc"),
        texcmd("usepackage").arg("array, tikz"),
        document(elems!(
            centering(elems!(texcmd("resizebox")
                .arg(texcmd("textwidth"))
                .arg("!")
                .arg(texenv(
                    "tikzpicture",
                    elems!(tikz_gridx, tikz_gridy, tikz_boundaries, tikz_clues)
                )))),
            texcmd("medskip"),
            raw("\n"),
            "Uppgiften är att fylla varje ruta med rätt uttryck på formen ",
            raw("$ax = b$"),
            " där varje koefficient är bland ",
            raw(tex_array(coeffs)),
            " och varje koefficient förekommer exakt en gång per rad och kolumn."
        ))
    ));

    tpl::TexElement::render(&tex)
        .expect("Failed to build .tex document. Source code contains invalid utf8 strings.")
}

// Returns byte representation of a pdf polyku rendered with latexmk.
pub(crate) fn pdf_doc(
    size: usize,
    coeffs: Vec<isize>,
    tiles: HashMap<usize, Vec<Node>>,
    clues: HashMap<usize, (String, String)>,
    boundaries: HashMap<usize, Vec<Node>>,
    ans: Vec<String>,
) -> Vec<u8> {
    texrender::TexRender::from_bytes(
        tex_doc(size, coeffs, tiles, clues, boundaries, ans).into_bytes(),
    )
    .render()
    .expect("Rendering error.")
}
