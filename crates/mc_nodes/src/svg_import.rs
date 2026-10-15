// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Path — SVG Import: an `.svg` file or a pasted
//! `d` string into the Path node's existing data model
//! ([`mc_core::Path`]/[`mc_core::SubPath`]/[`mc_core::Point`], unchanged).
//!
//! Ported from the owner's own reference package (`SVG_IMPORT_SPEC.md`,
//! `svg_import_reference.py` — the Python code is the ground truth; if the
//! spec and the code ever disagree, the code wins, per the package's own
//! README). Sources are the public W3C SVG 2 spec (path grammar, arc
//! endpoint-to-centre conversion, basic shapes, transforms) and the
//! standard "arc as a cubic" result — no Inkscape, Potrace, or Adobe code
//! referenced or copied. `roxmltree` walks the XML tree (its API maps
//! closely onto the reference's own use of Python's `ElementTree`); the
//! SVG path-data grammar itself — the part that has to match the
//! reference's own hand-rolled parser exactly — is hand-ported below, not
//! delegated to any library.
//!
//! Built one step at a time per the package's own README build order,
//! confirming each step's own test results before moving to the next:
//! 1. Path-data parser + point building + kind classification (done) —
//!    cases 01, 02, 03, 07, 12, 13.
//! 2. Arcs and basic shapes (done) — cases 04, 05, 08, 09.
//! 3. Transforms and fit modes (this file, right now) — cases 06, 10, 11.
//! 4. Warnings, robustness, then the node UI.
//!
//! Case 01 (step 1) already needed a `<polyline>` and a `translate`
//! transform, so `shape_to_d` already covered `polyline`/`polygon`, and
//! the transform matrix, the tree-walk's transform accumulator, and
//! `import_svg`'s three fit-mode branches were all already built (cheap,
//! self-contained, needed regardless) — this step is just the first
//! golden cases to actually exercise `rotate`/`scale`/`skewX`/`matrix`
//! composition and the `viewbox`/`none` fit modes.

use mc_core::{Path, Point, PointKind, SubPath};

// ---------------------------------------------------------------------
// 2D affine transforms (SVG order: x' = a*x + c*y + e, y' = b*x + d*y + f)
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct Transform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl Transform {
    const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    fn translate(x: f64, y: f64) -> Self {
        Self {
            e: x,
            f: y,
            ..Self::IDENTITY
        }
    }

    fn scale(x: f64, y: f64) -> Self {
        Self {
            a: x,
            d: y,
            ..Self::IDENTITY
        }
    }

    fn rotate(deg: f64) -> Self {
        let r = deg.to_radians();
        let (co, si) = (r.cos(), r.sin());
        Self {
            a: co,
            b: si,
            c: -si,
            d: co,
            ..Self::IDENTITY
        }
    }

    fn skew_x(deg: f64) -> Self {
        Self {
            c: deg.to_radians().tan(),
            ..Self::IDENTITY
        }
    }

    fn skew_y(deg: f64) -> Self {
        Self {
            b: deg.to_radians().tan(),
            ..Self::IDENTITY
        }
    }

    /// `self @ other` (SVG/numpy order): `other` is applied to the point
    /// first, `self` second.
    fn mul(self, other: Self) -> Self {
        Self {
            a: self.a * other.a + self.c * other.b,
            c: self.a * other.c + self.c * other.d,
            e: self.a * other.e + self.c * other.f + self.e,
            b: self.b * other.a + self.d * other.b,
            d: self.b * other.c + self.d * other.d,
            f: self.b * other.e + self.d * other.f + self.f,
        }
    }

    fn apply(self, p: (f64, f64)) -> (f64, f64) {
        (
            self.a * p.0 + self.c * p.1 + self.e,
            self.b * p.0 + self.d * p.1 + self.f,
        )
    }
}

/// SVG `transform` attribute list, applied left to right as written:
/// `"translate(..) rotate(..)"` means `M = T @ R` (the rightmost function
/// is applied to the point first) — ported from `parse_transform`.
fn parse_transform(s: Option<&str>) -> Transform {
    let Some(s) = s else {
        return Transform::IDENTITY;
    };
    let mut m = Transform::IDENTITY;
    let mut rest = s;
    while let Some(open) = rest.find('(') {
        let name = rest[..open].trim_end();
        let name = name
            .rsplit(|c: char| !c.is_alphabetic())
            .next()
            .unwrap_or(name);
        let Some(close) = rest[open..].find(')') else {
            break;
        };
        let args_str = &rest[open + 1..open + close];
        let v: Vec<f64> = extract_numbers(args_str);
        let t = match name {
            "matrix" if v.len() >= 6 => Transform {
                a: v[0],
                b: v[1],
                c: v[2],
                d: v[3],
                e: v[4],
                f: v[5],
            },
            "translate" if !v.is_empty() => {
                Transform::translate(v[0], v.get(1).copied().unwrap_or(0.0))
            }
            "scale" if !v.is_empty() => Transform::scale(v[0], v.get(1).copied().unwrap_or(v[0])),
            "rotate" if !v.is_empty() => {
                let r = Transform::rotate(v[0]);
                if v.len() >= 3 {
                    Transform::translate(v[1], v[2])
                        .mul(r)
                        .mul(Transform::translate(-v[1], -v[2]))
                } else {
                    r
                }
            }
            "skewX" if !v.is_empty() => Transform::skew_x(v[0]),
            "skewY" if !v.is_empty() => Transform::skew_y(v[0]),
            _ => Transform::IDENTITY,
        };
        m = m.mul(t);
        rest = &rest[open + close + 1..];
    }
    m
}

/// Every greedily-matched number in `s` (same `NUM_RE` the path-data
/// tokenizer uses), in order — used for `transform` argument lists and
/// `viewBox`/`points` attributes, which (unlike path data) have no
/// command letters or error semantics to worry about.
fn extract_numbers(s: &str) -> Vec<f64> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if let Some(len) = match_number(&chars[i..]) {
            let text: String = chars[i..i + len].iter().collect();
            if let Ok(v) = text.parse::<f64>() {
                out.push(v);
            }
            i += len;
        } else {
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------
// Path data ("d=...") -> absolute subpaths of L / C segments
// ---------------------------------------------------------------------

const CMD_CHARS: &str = "MmZzLlHhVvCcSsQqTtAa";

/// Greedily matches `NUM_RE = [-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?` at
/// the start of `chars`, returning how many chars it consumed, or `None`
/// if nothing there is a valid number. Hand-written rather than pulling in
/// a regex engine, since this also has to replicate the exact greedy
/// backtracking behaviour of the Python reference's own regex for compact
/// runs like `"1.5.5"` -> `1.5`, `.5` or `"1-2"` -> `1`, `-2`.
fn match_number(chars: &[char]) -> Option<usize> {
    let mut i = 0;
    if i < chars.len() && (chars[i] == '-' || chars[i] == '+') {
        i += 1;
    }
    let sign_end = i;
    let mut int_end = i;
    while int_end < chars.len() && chars[int_end].is_ascii_digit() {
        int_end += 1;
    }
    let has_int = int_end > sign_end;
    let mut mantissa_end = int_end;
    if mantissa_end < chars.len() && chars[mantissa_end] == '.' {
        let mut k = mantissa_end + 1;
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
        let has_frac_digits = k > mantissa_end + 1;
        if has_int || has_frac_digits {
            mantissa_end = k;
        }
    }
    if !has_int && mantissa_end == int_end {
        return None;
    }
    let mut end = mantissa_end;
    if end < chars.len() && (chars[end] == 'e' || chars[end] == 'E') {
        let mut j = end + 1;
        if j < chars.len() && (chars[j] == '-' || chars[j] == '+') {
            j += 1;
        }
        let exp_digit_start = j;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_digit_start {
            end = j;
        }
    }
    Some(end)
}

#[derive(Debug, Clone)]
enum Tok {
    Cmd(char),
    Num(String),
}

/// Commands and numbers, matching the Python reference's own `tokenize`:
/// `'1.5.5'` -> `1.5`, `.5`; `'1-2'` -> `1`, `-2`; whitespace/commas are
/// separators, skipped rather than emitted. On a character that's none of
/// command/number/separator, stops there and returns `true` (error) —
/// everything tokenized before it is still returned, per the W3C error
/// rule ("keep everything before the first error").
fn tokenize(d: &str) -> (Vec<Tok>, bool) {
    let chars: Vec<char> = d.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if CMD_CHARS.contains(c) {
            out.push(Tok::Cmd(c));
            i += 1;
        } else if let Some(len) = match_number(&chars[i..]) {
            out.push(Tok::Num(chars[i..i + len].iter().collect()));
            i += len;
        } else if c.is_whitespace() || c == ',' {
            while i < chars.len() && (chars[i].is_whitespace() || chars[i] == ',') {
                i += 1;
            }
        } else {
            return (out, true);
        }
    }
    (out, false)
}

type P2 = (f64, f64);

#[derive(Debug, Clone, Copy)]
enum Seg {
    L(P2, P2),
    C(P2, P2, P2, P2),
}

impl Seg {
    fn start(&self) -> P2 {
        match *self {
            Seg::L(p0, _) => p0,
            Seg::C(p0, ..) => p0,
        }
    }
    fn end(&self) -> P2 {
        match *self {
            Seg::L(_, p1) => p1,
            Seg::C(.., p1) => p1,
        }
    }
    fn transformed(&self, t: Transform) -> Seg {
        match *self {
            Seg::L(p0, p1) => Seg::L(t.apply(p0), t.apply(p1)),
            Seg::C(p0, c1, c2, p1) => Seg::C(t.apply(p0), t.apply(c1), t.apply(c2), t.apply(p1)),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Sub {
    segs: Vec<Seg>,
    closed: bool,
}

/// W3C SVG 2 Implementation Notes B.2.4 (endpoint-to-centre conversion,
/// out-of-range radius scale-up), then split into `n = ceil(|dtheta| /
/// 90deg)` equal pieces, each a cubic with handle factor `k = 4/3 *
/// tan(delta / 4)` — ported 1:1 from `arc_to_cubics`.
#[allow(clippy::too_many_arguments)]
fn arc_to_cubics(p1: P2, rx: f64, ry: f64, phi_deg: f64, fa: bool, fs: bool, p2: P2) -> Vec<Seg> {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    if (x1 - x2).abs() < 1e-12 && (y1 - y2).abs() < 1e-12 {
        return Vec::new(); // omitted
    }
    let (mut rx, mut ry) = (rx.abs(), ry.abs());
    if rx < 1e-12 || ry < 1e-12 {
        return vec![Seg::L(p1, p2)];
    }
    let phi = phi_deg.rem_euclid(360.0).to_radians();
    let (co, si) = (phi.cos(), phi.sin());
    let (dx2, dy2) = ((x1 - x2) / 2.0, (y1 - y2) / 2.0);
    let x1p = co * dx2 + si * dy2;
    let y1p = -si * dx2 + co * dy2;
    let lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry);
    if lam > 1.0 {
        let s = lam.sqrt();
        rx *= s;
        ry *= s;
    }
    let num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    let den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    let mut coef = if den > 0.0 {
        (num / den).max(0.0).sqrt()
    } else {
        0.0
    };
    if fa == fs {
        coef = -coef;
    }
    let cxp = coef * rx * y1p / ry;
    let cyp = -coef * ry * x1p / rx;
    let cx = co * cxp - si * cyp + (x1 + x2) / 2.0;
    let cy = si * cxp + co * cyp + (y1 + y2) / 2.0;

    let ang = |ux: f64, uy: f64, vx: f64, vy: f64| (ux * vy - uy * vx).atan2(ux * vx + uy * vy);
    let (ux, uy) = ((x1p - cxp) / rx, (y1p - cyp) / ry);
    let (vx, vy) = ((-x1p - cxp) / rx, (-y1p - cyp) / ry);
    let th1 = ang(1.0, 0.0, ux, uy);
    let mut dth = ang(ux, uy, vx, vy);
    if !fs && dth > 0.0 {
        dth -= 2.0 * std::f64::consts::PI;
    } else if fs && dth < 0.0 {
        dth += 2.0 * std::f64::consts::PI;
    }
    let n = (1i64).max(((dth.abs() / (std::f64::consts::PI / 2.0)) - 1e-9).ceil() as i64);
    let d = dth / n as f64;
    let k = 4.0 / 3.0 * (d / 4.0).tan();

    let e = |t: f64| -> P2 {
        (
            cx + rx * t.cos() * co - ry * t.sin() * si,
            cy + rx * t.cos() * si + ry * t.sin() * co,
        )
    };
    let ed = |t: f64| -> P2 {
        (
            -rx * t.sin() * co - ry * t.cos() * si,
            -rx * t.sin() * si + ry * t.cos() * co,
        )
    };

    let mut out = Vec::with_capacity(n as usize);
    let mut t0 = th1;
    let mut prev = (x1, y1);
    for i in 0..n {
        let t1 = t0 + d;
        let q0 = prev;
        let q3 = if i == n - 1 { (x2, y2) } else { e(t1) };
        let (d0, d1) = (ed(t0), ed(t1));
        let c1 = (q0.0 + k * d0.0, q0.1 + k * d0.1);
        let c2 = (q3.0 - k * d1.0, q3.1 - k * d1.1);
        out.push(Seg::C(q0, c1, c2, q3));
        prev = q3;
        t0 = t1;
    }
    out
}

struct ParseState {
    toks: Vec<Tok>,
    i: usize,
    cur: P2,
    start: P2,
    cmd: Option<char>,
    last_c2: Option<P2>,
    last_q: Option<P2>,
    sub: Option<usize>,
    subs: Vec<Sub>,
}

struct ParseError;

impl ParseState {
    fn is_cmd_at(&self, k: usize) -> bool {
        matches!(self.toks.get(k), Some(Tok::Cmd(_)))
    }

    fn num(&mut self) -> Result<f64, ParseError> {
        if self.i >= self.toks.len() || self.is_cmd_at(self.i) {
            return Err(ParseError);
        }
        let Tok::Num(s) = &self.toks[self.i] else {
            return Err(ParseError);
        };
        let v: f64 = s.parse().map_err(|_| ParseError)?;
        self.i += 1;
        Ok(v)
    }

    fn flag(&mut self) -> Result<bool, ParseError> {
        if self.i >= self.toks.len() {
            return Err(ParseError);
        }
        let Tok::Num(s) = &mut self.toks[self.i] else {
            return Err(ParseError);
        };
        let first = s.chars().next().ok_or(ParseError)?;
        if first != '0' && first != '1' {
            return Err(ParseError);
        }
        if s.len() == 1 {
            self.i += 1;
        } else {
            *s = s[first.len_utf8()..].to_string();
        }
        Ok(first == '1')
    }

    fn ensure(&mut self) -> usize {
        if self.sub.is_none() {
            self.subs.push(Sub {
                segs: Vec::new(),
                closed: false,
            });
            self.sub = Some(self.subs.len() - 1);
        }
        self.sub.unwrap()
    }
}

/// SVG path grammar -> absolute subpaths (W3C SVG 2 Paths) — ported
/// 1:1 from the Python reference's own `parse_d`. Appends a warning if the
/// data has an error (bad character, or an incomplete command dropped at
/// the end); everything parsed before the error is kept either way.
fn parse_d(d: &str, warnings: &mut Vec<String>) -> Vec<Sub> {
    let (toks, bad) = tokenize(d);
    let mut st = ParseState {
        toks,
        i: 0,
        cur: (0.0, 0.0),
        start: (0.0, 0.0),
        cmd: None,
        last_c2: None,
        last_q: None,
        sub: None,
        subs: Vec::new(),
    };
    let mut error = bad;

    let result: Result<(), ParseError> = (|| {
        while st.i < st.toks.len() {
            if let Tok::Cmd(c) = st.toks[st.i] {
                st.cmd = Some(c);
                st.i += 1;
            } else if st.cmd.is_none() {
                return Err(ParseError);
            }
            let cmd = st.cmd.unwrap();
            let rel = cmd.is_ascii_lowercase();
            let c_upper = cmd.to_ascii_uppercase();
            let cur = st.cur;
            let (ox, oy) = if rel { cur } else { (0.0, 0.0) };

            if c_upper == 'Z' {
                let si = st.ensure();
                if cur != st.start {
                    st.subs[si].segs.push(Seg::L(cur, st.start));
                }
                st.subs[si].closed = true;
                st.cur = st.start;
                st.sub = None;
                st.last_c2 = None;
                st.last_q = None;
                st.cmd = None;
                continue;
            }
            if c_upper == 'M' {
                let p = (ox + st.num()?, oy + st.num()?);
                st.subs.push(Sub {
                    segs: Vec::new(),
                    closed: false,
                });
                st.sub = Some(st.subs.len() - 1);
                st.cur = p;
                st.start = p;
                st.last_c2 = None;
                st.last_q = None;
                st.cmd = Some(if cmd == 'M' { 'L' } else { 'l' });
                continue;
            }

            // Read all arguments first, so an incomplete command adds nothing.
            let args: Vec<f64> = match c_upper {
                'L' => vec![st.num()?, st.num()?],
                'H' | 'V' => vec![st.num()?],
                'C' => {
                    let mut a = Vec::with_capacity(6);
                    for _ in 0..6 {
                        a.push(st.num()?);
                    }
                    a
                }
                'S' | 'Q' => {
                    let mut a = Vec::with_capacity(4);
                    for _ in 0..4 {
                        a.push(st.num()?);
                    }
                    a
                }
                'T' => vec![st.num()?, st.num()?],
                'A' => {
                    let rx = st.num()?;
                    let ry = st.num()?;
                    let rot = st.num()?;
                    let large_arc = st.flag()?;
                    let sweep = st.flag()?;
                    let x = st.num()?;
                    let y = st.num()?;
                    vec![
                        rx,
                        ry,
                        rot,
                        f64::from(large_arc as u8),
                        f64::from(sweep as u8),
                        x,
                        y,
                    ]
                }
                _ => unreachable!("not a path command"),
            };

            let si = st.ensure();
            let lc2 = st.last_c2.take();
            let lq = st.last_q.take();

            let new_cur = match c_upper {
                'L' => {
                    let p = (ox + args[0], oy + args[1]);
                    st.subs[si].segs.push(Seg::L(cur, p));
                    p
                }
                'H' => {
                    let p = (ox + args[0], cur.1);
                    st.subs[si].segs.push(Seg::L(cur, p));
                    p
                }
                'V' => {
                    let p = (cur.0, oy + args[0]);
                    st.subs[si].segs.push(Seg::L(cur, p));
                    p
                }
                'C' => {
                    let c1 = (ox + args[0], oy + args[1]);
                    let c2 = (ox + args[2], oy + args[3]);
                    let p = (ox + args[4], oy + args[5]);
                    st.subs[si].segs.push(Seg::C(cur, c1, c2, p));
                    st.last_c2 = Some(c2);
                    p
                }
                'S' => {
                    let c1 = match lc2 {
                        Some(lc2) => (2.0 * cur.0 - lc2.0, 2.0 * cur.1 - lc2.1),
                        None => cur,
                    };
                    let c2 = (ox + args[0], oy + args[1]);
                    let p = (ox + args[2], oy + args[3]);
                    st.subs[si].segs.push(Seg::C(cur, c1, c2, p));
                    st.last_c2 = Some(c2);
                    p
                }
                'Q' | 'T' => {
                    let (q, p) = if c_upper == 'Q' {
                        ((ox + args[0], oy + args[1]), (ox + args[2], oy + args[3]))
                    } else {
                        let q = match lq {
                            Some(lq) => (2.0 * cur.0 - lq.0, 2.0 * cur.1 - lq.1),
                            None => cur,
                        };
                        (q, (ox + args[0], oy + args[1]))
                    };
                    let c1 = (
                        cur.0 + 2.0 / 3.0 * (q.0 - cur.0),
                        cur.1 + 2.0 / 3.0 * (q.1 - cur.1),
                    );
                    let c2 = (p.0 + 2.0 / 3.0 * (q.0 - p.0), p.1 + 2.0 / 3.0 * (q.1 - p.1));
                    st.subs[si].segs.push(Seg::C(cur, c1, c2, p));
                    st.last_q = Some(q);
                    p
                }
                'A' => {
                    let p = (ox + args[5], oy + args[6]);
                    let arc = arc_to_cubics(
                        cur,
                        args[0],
                        args[1],
                        args[2],
                        args[3] != 0.0,
                        args[4] != 0.0,
                        p,
                    );
                    st.subs[si].segs.extend(arc);
                    p
                }
                _ => unreachable!(),
            };
            st.cur = new_cur;
        }
        Ok(())
    })();

    if result.is_err() {
        error = true;
    }
    if error {
        warnings.push("path data error: kept everything before it".to_string());
    }
    st.subs.into_iter().filter(|s| !s.segs.is_empty()).collect()
}

// ---------------------------------------------------------------------
// Basic shapes -> path data (W3C SVG 2 "Basic Shapes")
// ---------------------------------------------------------------------

fn shape_to_d(node: roxmltree::Node, tag: &str) -> Option<String> {
    match tag {
        "path" => Some(node.attribute("d").unwrap_or("").to_string()),
        "rect" => {
            let x = attr_f64(node, "x", 0.0);
            let y = attr_f64(node, "y", 0.0);
            let w = attr_f64(node, "width", 0.0);
            let h = attr_f64(node, "height", 0.0);
            if w <= 0.0 || h <= 0.0 {
                return None;
            }
            let rx_set = node.attribute("rx").is_some();
            let ry_set = node.attribute("ry").is_some();
            let (mut rx, mut ry) = match (rx_set, ry_set) {
                (false, false) => (0.0, 0.0),
                (true, false) => {
                    let rx = attr_f64(node, "rx", 0.0);
                    (rx, rx)
                }
                (false, true) => {
                    let ry = attr_f64(node, "ry", 0.0);
                    (ry, ry)
                }
                (true, true) => (attr_f64(node, "rx", 0.0), attr_f64(node, "ry", 0.0)),
            };
            rx = rx.abs().min(w / 2.0);
            ry = ry.abs().min(h / 2.0);
            if rx == 0.0 || ry == 0.0 {
                return Some(format!("M{x},{y} H{} V{} H{x} Z", x + w, y + h));
            }
            Some(format!(
                "M{},{y} H{} A{rx},{ry} 0 0 1 {},{} V{} A{rx},{ry} 0 0 1 {},{} H{} A{rx},{ry} 0 0 1 {x},{} V{} A{rx},{ry} 0 0 1 {},{y} Z",
                x + rx,
                x + w - rx,
                x + w,
                y + ry,
                y + h - ry,
                x + w - rx,
                y + h,
                x + rx,
                y + h - ry,
                y + ry,
                x + rx,
            ))
        }
        "circle" | "ellipse" => {
            let cx = attr_f64(node, "cx", 0.0);
            let cy = attr_f64(node, "cy", 0.0);
            let (rx, ry) = if tag == "circle" {
                let r = attr_f64(node, "r", 0.0);
                (r, r)
            } else {
                (attr_f64(node, "rx", 0.0), attr_f64(node, "ry", 0.0))
            };
            if rx <= 0.0 || ry <= 0.0 {
                return None;
            }
            // Starts at (cx + rx, cy), four quarter arcs clockwise on screen.
            Some(format!(
                "M{},{cy} A{rx},{ry} 0 0 1 {cx},{} A{rx},{ry} 0 0 1 {},{cy} A{rx},{ry} 0 0 1 {cx},{} A{rx},{ry} 0 0 1 {},{cy} Z",
                cx + rx,
                cy + ry,
                cx - rx,
                cy - ry,
                cx + rx,
            ))
        }
        "line" => Some(format!(
            "M{},{} L{},{}",
            attr_f64(node, "x1", 0.0),
            attr_f64(node, "y1", 0.0),
            attr_f64(node, "x2", 0.0),
            attr_f64(node, "y2", 0.0),
        )),
        "polyline" | "polygon" => {
            let v = extract_numbers(node.attribute("points").unwrap_or(""));
            if v.len() < 4 {
                return None;
            }
            let pairs = v.len() / 2 * 2;
            let mut d = String::from("M");
            for k in (0..pairs).step_by(2) {
                if k > 0 {
                    d.push_str(" L");
                }
                d.push_str(&format!("{},{}", v[k], v[k + 1]));
            }
            if tag == "polygon" {
                d.push_str(" Z");
            }
            Some(d)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------
// Tree walk
// ---------------------------------------------------------------------

const SHAPES: &[&str] = &[
    "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
];
const CONTAINERS: &[&str] = &["svg", "g", "a", "switch"];
const SKIP: &[&str] = &[
    "defs",
    "clipPath",
    "mask",
    "symbol",
    "pattern",
    "marker",
    "style",
    "title",
    "desc",
    "metadata",
    "linearGradient",
    "radialGradient",
    "filter",
];

fn attr_f64(node: roxmltree::Node, name: &str, default: f64) -> f64 {
    let Some(v) = node.attribute(name) else {
        return default;
    };
    let chars: Vec<char> = v.trim().chars().collect();
    match match_number(&chars) {
        Some(len) => chars[..len]
            .iter()
            .collect::<String>()
            .parse()
            .unwrap_or(default),
        None => default,
    }
}

fn is_hidden(node: roxmltree::Node) -> bool {
    if node.attribute("display") == Some("none") || node.attribute("visibility") == Some("hidden") {
        return true;
    }
    let Some(style) = node.attribute("style") else {
        return false;
    };
    for decl in style.split(';') {
        let Some((k, v)) = decl.split_once(':') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        if (k == "display" && v == "none") || (k == "visibility" && v == "hidden") {
            return true;
        }
    }
    false
}

struct Collected {
    items: Vec<(Sub, Transform)>,
    warnings: Vec<String>,
    element_count: u32,
}

const XLINK_NS: &str = "http://www.w3.org/1999/xlink";

fn href_of<'a>(node: roxmltree::Node<'a, 'a>) -> Option<&'a str> {
    node.attribute("href")
        .or_else(|| node.attribute((XLINK_NS, "href")))
}

fn collect(root: roxmltree::Node) -> Collected {
    let ids: std::collections::HashMap<&str, roxmltree::Node> = root
        .descendants()
        .filter(|n| n.is_element())
        .filter_map(|n| n.attribute("id").map(|id| (id, n)))
        .collect();

    let mut items = Vec::new();
    let mut warnings = Vec::new();
    let mut element_count = 0u32;

    fn walk(
        node: roxmltree::Node,
        m: Transform,
        depth: u32,
        ids: &std::collections::HashMap<&str, roxmltree::Node>,
        items: &mut Vec<(Sub, Transform)>,
        warnings: &mut Vec<String>,
        element_count: &mut u32,
    ) {
        if depth > 32 {
            warnings.push("use/nesting depth > 32, stopped".to_string());
            return;
        }
        let tag = node.tag_name().name();
        if is_hidden(node) {
            return;
        }
        if SKIP.contains(&tag) {
            return;
        }
        let mut m2 = m.mul(parse_transform(node.attribute("transform")));
        if CONTAINERS.contains(&tag) {
            if tag == "svg" && depth > 0 {
                // Nested <svg>: offset only, viewBox ignored.
                m2 = m2.mul(Transform::translate(
                    attr_f64(node, "x", 0.0),
                    attr_f64(node, "y", 0.0),
                ));
            }
            for child in node.children().filter(|n| n.is_element()) {
                walk(child, m2, depth + 1, ids, items, warnings, element_count);
            }
        } else if tag == "use" {
            let href = href_of(node).unwrap_or("");
            let Some(&reference) = ids.get(href.trim_start_matches('#')) else {
                warnings.push(format!("use: missing {href}"));
                return;
            };
            let m3 = m2.mul(Transform::translate(
                attr_f64(node, "x", 0.0),
                attr_f64(node, "y", 0.0),
            ));
            if reference.tag_name().name() == "symbol" {
                for child in reference.children().filter(|n| n.is_element()) {
                    walk(child, m3, depth + 1, ids, items, warnings, element_count);
                }
            } else {
                walk(
                    reference,
                    m3,
                    depth + 1,
                    ids,
                    items,
                    warnings,
                    element_count,
                );
            }
        } else if SHAPES.contains(&tag) {
            let Some(d) = shape_to_d(node, tag) else {
                return;
            };
            if d.is_empty() {
                return;
            }
            *element_count += 1;
            let mut local_warnings = Vec::new();
            for s in parse_d(&d, &mut local_warnings) {
                items.push((s, m2));
            }
            warnings.extend(local_warnings);
        } else {
            warnings.push(format!("ignored <{tag}>"));
        }
    }

    walk(
        root,
        Transform::IDENTITY,
        0,
        &ids,
        &mut items,
        &mut warnings,
        &mut element_count,
    );
    warnings.sort();
    warnings.dedup();
    Collected {
        items,
        warnings,
        element_count,
    }
}

// ---------------------------------------------------------------------
// Segments -> Path points
// ---------------------------------------------------------------------

const EPS_MERGE: f64 = 1e-9;
const SMOOTH_DEG: f64 = 2.0;
const SYMMETRIC_REL: f64 = 0.01;

fn classify(h_in: P2, h_out: P2) -> PointKind {
    let li = (h_in.0 * h_in.0 + h_in.1 * h_in.1).sqrt();
    let lo = (h_out.0 * h_out.0 + h_out.1 * h_out.1).sqrt();
    if li < 1e-12 || lo < 1e-12 {
        return PointKind::Corner;
    }
    let cosang = -(h_in.0 * h_out.0 + h_in.1 * h_out.1) / (li * lo);
    if cosang < SMOOTH_DEG.to_radians().cos() {
        return PointKind::Corner;
    }
    if (li - lo).abs() <= SYMMETRIC_REL * li.max(lo) {
        PointKind::Symmetric
    } else {
        PointKind::Smooth
    }
}

/// Each segment end becomes a Point; consecutive segments share their
/// joint. A closed subpath whose last joint coincides with its start
/// (within `EPS_MERGE * scale`) has the two merged (no zero-length
/// closing segment) — ported from `sub_to_points`.
fn sub_to_points(s: &Sub, scale: f64) -> SubPath {
    struct Raw {
        pos: P2,
        h_in: P2,
        h_out: P2,
    }
    let mut pts: Vec<Raw> = Vec::new();
    let push_pos = |pts: &mut Vec<Raw>, pos: P2| {
        pts.push(Raw {
            pos,
            h_in: (0.0, 0.0),
            h_out: (0.0, 0.0),
        });
    };
    push_pos(&mut pts, s.segs[0].start());
    for g in &s.segs {
        let p1 = g.end();
        if let Seg::C(p0, c1, c2, _) = *g {
            pts.last_mut().unwrap().h_out = (c1.0 - p0.0, c1.1 - p0.1);
            push_pos(&mut pts, p1);
            let n = pts.len();
            pts[n - 1].h_in = (c2.0 - p1.0, c2.1 - p1.1);
        } else {
            push_pos(&mut pts, p1);
        }
    }
    let mut closed = s.closed;
    if closed {
        let a = pts[0].pos;
        let b = pts[pts.len() - 1].pos;
        let dist = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        if dist <= EPS_MERGE * scale && pts.len() > 1 {
            let last_h_in = pts.last().unwrap().h_in;
            pts[0].h_in = last_h_in;
            pts.pop();
        }
    }
    if pts.len() < 2 {
        closed = false;
    }
    let n = pts.len();
    let points: Vec<Point> = pts
        .iter()
        .enumerate()
        .map(|(k, raw)| {
            let kind = if !closed && (k == 0 || k == n - 1) {
                PointKind::Corner
            } else {
                classify(raw.h_in, raw.h_out)
            };
            Point {
                pos: [raw.pos.0, raw.pos.1],
                h_in: [raw.h_in.0, raw.h_in.1],
                h_out: [raw.h_out.0, raw.h_out.1],
                kind,
                width: 1.0,
            }
        })
        .collect();
    let closed = closed && points.len() >= 2;
    SubPath::new(points, closed)
}

/// Tight bbox of the actual curves — cubic extrema solved exactly (not
/// just control-point bounds), ported from `_bbox_of_subs`.
fn bbox_of_subs(subs: &[Sub]) -> (f64, f64, f64, f64) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for s in subs {
        for g in &s.segs {
            xs.push(g.start().0);
            xs.push(g.end().0);
            ys.push(g.start().1);
            ys.push(g.end().1);
            if let Seg::C(p0, p1, p2, p3) = *g {
                for (axis, lst) in [(0usize, &mut xs), (1usize, &mut ys)] {
                    let pick = |p: P2| if axis == 0 { p.0 } else { p.1 };
                    let (c0, c1, c2, c3) = (pick(p0), pick(p1), pick(p2), pick(p3));
                    let a = -c0 + 3.0 * c1 - 3.0 * c2 + c3;
                    let b = 2.0 * (c0 - 2.0 * c1 + c2);
                    let c = c1 - c0;
                    let mut roots = Vec::new();
                    if a.abs() < 1e-14 {
                        if b.abs() > 1e-14 {
                            roots.push(-c / b);
                        }
                    } else {
                        let disc = b * b - 4.0 * a * c;
                        if disc >= 0.0 {
                            let sq = disc.sqrt();
                            roots.push((-b + sq) / (2.0 * a));
                            roots.push((-b - sq) / (2.0 * a));
                        }
                    }
                    for t in roots {
                        if t > 0.0 && t < 1.0 {
                            let mt = 1.0 - t;
                            let v = mt.powi(3) * c0
                                + 3.0 * mt * mt * t * c1
                                + 3.0 * mt * t * t * c2
                                + t.powi(3) * c3;
                            lst.push(v);
                        }
                    }
                }
            }
        }
    }
    let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min_x, min_y, max_x, max_y)
}

fn parse_viewbox(root: roxmltree::Node) -> Option<(f64, f64, f64, f64)> {
    if let Some(vb) = root.attribute("viewBox") {
        let v = extract_numbers(vb);
        if v.len() == 4 && v[2] > 0.0 && v[3] > 0.0 {
            return Some((v[0], v[1], v[2], v[3]));
        }
    }
    let w = attr_f64(root, "width", 0.0);
    let h = attr_f64(root, "height", 0.0);
    if w > 0.0 && h > 0.0 {
        return Some((0.0, 0.0, w, h));
    }
    None
}

// ---------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    Bbox,
    Viewbox,
    None,
}

pub struct ImportResult {
    pub path: Path,
    pub warnings: Vec<String>,
    pub viewbox: Option<(f64, f64, f64, f64)>,
    pub source_bbox: (f64, f64, f64, f64),
    pub element_count: u32,
}

/// `fit`: `Bbox` scales the drawing's tight bbox uniformly into
/// `[margin, 1 - margin]^2`, centred (the default). `Viewbox` maps the
/// viewBox (or width/height) to the tile like `xMidYMid meet` (margin
/// ignored), falling back to `Bbox` if there's none. `None` treats 1 user
/// unit as 1 tile unit. Ported from `import_svg`.
pub fn import_svg(svg_text: &str, fit: Fit, margin: f64, wrap: bool) -> ImportResult {
    let doc = match roxmltree::Document::parse(svg_text) {
        Ok(doc) => doc,
        Err(_) => {
            return ImportResult {
                path: Path {
                    subpaths: Vec::new(),
                    wrap,
                },
                warnings: vec!["could not parse XML".to_string()],
                viewbox: None,
                source_bbox: (0.0, 0.0, 0.0, 0.0),
                element_count: 0,
            };
        }
    };
    let root = doc.root_element();
    let collected = collect(root);
    let subs: Vec<Sub> = collected
        .items
        .iter()
        .map(|(s, m)| Sub {
            segs: s.segs.iter().map(|g| g.transformed(*m)).collect(),
            closed: s.closed,
        })
        .collect();
    let viewbox = parse_viewbox(root);

    if subs.is_empty() {
        let mut warnings = collected.warnings;
        warnings.push("no geometry".to_string());
        return ImportResult {
            path: Path {
                subpaths: Vec::new(),
                wrap,
            },
            warnings,
            viewbox,
            source_bbox: (0.0, 0.0, 0.0, 0.0),
            element_count: collected.element_count,
        };
    }

    let bb = bbox_of_subs(&subs);
    let (s, ox, oy) = match fit {
        Fit::Viewbox if viewbox.is_some() => {
            let (x0, y0, w, h) = viewbox.unwrap();
            let s = (1.0 / w).min(1.0 / h);
            (
                (s),
                (1.0 - w * s) / 2.0 - x0 * s,
                (1.0 - h * s) / 2.0 - y0 * s,
            )
        }
        Fit::None => (1.0, 0.0, 0.0),
        _ => {
            let w = (bb.2 - bb.0).max(1e-12);
            let h = (bb.3 - bb.1).max(1e-12);
            let avail = 1.0 - 2.0 * margin;
            let s = avail / w.max(h);
            (
                s,
                0.5 - (bb.0 + bb.2) / 2.0 * s,
                0.5 - (bb.1 + bb.3) / 2.0 * s,
            )
        }
    };
    let fit_transform = Transform {
        a: s,
        b: 0.0,
        c: 0.0,
        d: s,
        e: ox,
        f: oy,
    };
    let size = (bb.2 - bb.0).max(bb.3 - bb.1).max(1e-12);
    let scale = size * s;
    let subpaths: Vec<SubPath> = subs
        .iter()
        .map(|sb| Sub {
            segs: sb
                .segs
                .iter()
                .map(|g| g.transformed(fit_transform))
                .collect(),
            closed: sb.closed,
        })
        .map(|sb| sub_to_points(&sb, scale))
        .filter(|sp| sp.points.len() >= 2)
        .collect();

    ImportResult {
        path: Path { subpaths, wrap },
        warnings: collected.warnings,
        viewbox,
        source_bbox: bb,
        element_count: collected.element_count,
    }
}

/// Pasted `d` string (no XML) — wraps it in a minimal `<svg><path d="..."/>
/// </svg>` and reuses [`import_svg`].
pub fn import_d(d: &str, fit: Fit, margin: f64, wrap: bool) -> ImportResult {
    let escaped = d.replace('&', "&amp;").replace('"', "&quot;");
    let svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg"><path d="{escaped}"/></svg>"#);
    import_svg(&svg, fit, margin, wrap)
}

#[cfg(test)]
mod tests {
    use super::{import_d, import_svg, Fit};

    fn golden_path(file_name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/svg_import")
            .join(file_name)
    }

    fn assert_matches_golden(case: &str, result: &super::ImportResult) {
        let golden_text = std::fs::read_to_string(golden_path(&format!("{case}.json")))
            .unwrap_or_else(|err| panic!("failed to read {case}.json: {err}"));
        let golden: serde_json::Value = serde_json::from_str(&golden_text).unwrap();
        let golden_path_json = &golden["path"];
        let golden_subpaths = golden_path_json["subpaths"].as_array().unwrap();

        assert_eq!(
            result.path.subpaths.len(),
            golden_subpaths.len(),
            "{case}: subpath count"
        );
        for (si, (got_sp, want_sp)) in result.path.subpaths.iter().zip(golden_subpaths).enumerate()
        {
            let want_points = want_sp["points"].as_array().unwrap();
            assert_eq!(
                got_sp.points.len(),
                want_points.len(),
                "{case} subpath {si}: point count"
            );
            assert_eq!(
                got_sp.closed,
                want_sp["closed"].as_bool().unwrap(),
                "{case} subpath {si}: closed"
            );
            for (pi, (got_p, want_p)) in got_sp.points.iter().zip(want_points).enumerate() {
                let want_kind = want_p["kind"].as_str().unwrap();
                let got_kind = match got_p.kind {
                    mc_core::PointKind::Corner => "corner",
                    mc_core::PointKind::Smooth => "smooth",
                    mc_core::PointKind::Symmetric => "symmetric",
                };
                assert_eq!(got_kind, want_kind, "{case} subpath {si} point {pi}: kind");
                for (field, got_v, want_v) in [
                    ("pos.x", got_p.pos[0], want_p["pos"][0].as_f64().unwrap()),
                    ("pos.y", got_p.pos[1], want_p["pos"][1].as_f64().unwrap()),
                    ("h_in.x", got_p.h_in[0], want_p["h_in"][0].as_f64().unwrap()),
                    ("h_in.y", got_p.h_in[1], want_p["h_in"][1].as_f64().unwrap()),
                    (
                        "h_out.x",
                        got_p.h_out[0],
                        want_p["h_out"][0].as_f64().unwrap(),
                    ),
                    (
                        "h_out.y",
                        got_p.h_out[1],
                        want_p["h_out"][1].as_f64().unwrap(),
                    ),
                ] {
                    assert!(
                        (got_v - want_v).abs() < 1e-9,
                        "{case} subpath {si} point {pi} ({field}): {got_v}, expected {want_v}"
                    );
                }
            }
        }
        let want_warnings: Vec<String> = golden["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_str().unwrap().to_string())
            .collect();
        assert_eq!(result.warnings, want_warnings, "{case}: warnings");
    }

    /// Step 1 of the build order: path-data parsing, point building and
    /// kind classification, exact (positions/handles within 1e-9, same
    /// point count/closed/kind) against cases 01, 02, 03, 07 and 13.
    #[test]
    fn matches_the_python_reference_for_step_1_file_cases() {
        for case in [
            "01_lines_abs_rel",
            "02_cubic_smooth",
            "03_quadratic",
            "07_holes_and_closing",
            "13_errors_and_unfilled",
        ] {
            let svg_text = std::fs::read_to_string(golden_path(&format!("{case}.svg")))
                .unwrap_or_else(|err| panic!("failed to read {case}.svg: {err}"));
            let result = import_svg(&svg_text, Fit::Bbox, 0.05, true);
            assert_matches_golden(case, &result);
        }
    }

    /// Case 12: the existing Path-node paste feature (a bare `d` string,
    /// no XML).
    #[test]
    fn matches_the_python_reference_for_pasted_d_string() {
        let result = import_d("M0 0C10-20 30-20 40 0S70 20 80 0", Fit::Bbox, 0.05, true);
        assert_matches_golden("12_paste_d", &result);
    }

    /// Step 2 of the build order: arcs and basic shapes, cases 04, 05, 08,
    /// 09. The spec's own stated bar for this step is a looser one
    /// (symmetric Hausdorff distance between sampled curves <= 1e-4 tile,
    /// since a library's own arc-splitting convention might not match
    /// this port's) — but since `arc_to_cubics` here is a direct 1:1 port
    /// of the Python reference's own 90-degree split (not a third-party
    /// library's own arc conversion), it hits the exact same bar step 1
    /// used (same point counts/closed/kind, positions and handles within
    /// 1e-9) across all four cases, confirmed directly rather than
    /// assumed. That's strictly stronger than the Hausdorff bound (exact
    /// position agreement implies ~0 Hausdorff distance), so this reuses
    /// `assert_matches_golden` rather than also building separate
    /// `_samples.npy`-based Hausdorff-distance test machinery for a
    /// weaker check this port doesn't need.
    #[test]
    fn matches_the_python_reference_for_step_2_cases() {
        for case in [
            "04_arcs",
            "05_basic_shapes",
            "08_editor_export",
            "09_ornament",
        ] {
            let svg_text = std::fs::read_to_string(golden_path(&format!("{case}.svg")))
                .unwrap_or_else(|err| panic!("failed to read {case}.svg: {err}"));
            let result = import_svg(&svg_text, Fit::Bbox, 0.05, true);
            assert_matches_golden(case, &result);
        }
    }

    /// Step 3 of the build order: transforms and fit modes, exact (1e-9).
    /// Case 06 exercises nested groups with `translate`, `rotate`,
    /// `rotate(a cx cy)`, `scale`, `skewX` and `matrix`, composed left to
    /// right; cases 10/11 are the `viewbox`/`none` fit modes (both already
    /// built during steps 1-2, since `import_svg`'s fit-mode branches and
    /// the tree walk's transform accumulator needed to exist from the
    /// start — only now actually exercised by a golden case).
    #[test]
    fn matches_the_python_reference_for_step_3_cases() {
        let cases: &[(&str, &str, Fit)] = &[
            ("06_transforms", "06_transforms", Fit::Bbox),
            ("10_fit_viewbox", "05_basic_shapes", Fit::Viewbox),
            ("11_fit_none", "11_tile_units", Fit::None),
        ];
        for (golden_case, source_case, fit) in cases.iter().copied() {
            let svg_text = std::fs::read_to_string(golden_path(&format!("{source_case}.svg")))
                .unwrap_or_else(|err| panic!("failed to read {source_case}.svg: {err}"));
            let result = import_svg(&svg_text, fit, 0.05, true);
            assert_matches_golden(golden_case, &result);
        }
    }

    /// Step 4 of the build order, robustness half: "Import must never
    /// crash on any file in test_svgs/." Every `.svg` fixture this port's
    /// own golden tests already use, re-run here in every `Fit` mode (not
    /// just the one each golden case happens to use) — a plain "did it
    /// panic" sweep, independent of the exact-match assertions above.
    #[test]
    fn never_panics_on_any_test_svg_in_any_fit_mode() {
        let dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/svg_import");
        let mut found_any = false;
        for entry in std::fs::read_dir(&dir).expect("golden dir exists") {
            let path = entry.expect("dir entry readable").path();
            if path.extension().and_then(|e| e.to_str()) != Some("svg") {
                continue;
            }
            found_any = true;
            let svg_text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
            for fit in [Fit::Bbox, Fit::Viewbox, Fit::None] {
                for wrap in [true, false] {
                    let _ = import_svg(&svg_text, fit, 0.05, wrap);
                }
            }
        }
        assert!(found_any, "no .svg fixtures found in {}", dir.display());
    }

    /// Malformed/degenerate inputs the golden package's own svgs don't
    /// happen to cover: not XML at all, an empty string, and a `d` that's
    /// nothing but whitespace. None of these are a real SVG a user would
    /// ever import, but a paste/drag-and-drop UI will see exactly this
    /// kind of input eventually, and the spec's own rule is explicit:
    /// "Import must never crash" — plus "an empty file gives an empty
    /// path plus a 'no geometry' warning."
    #[test]
    fn never_panics_on_malformed_or_empty_input() {
        for bad in [
            "",
            "not xml at all {}<>",
            "<svg>",
            "<svg><path d=\"???\"/></svg>",
        ] {
            let result = import_svg(bad, Fit::Bbox, 0.05, true);
            assert!(
                result.path.subpaths.is_empty(),
                "malformed input {bad:?} unexpectedly produced geometry"
            );
            assert!(
                !result.warnings.is_empty(),
                "malformed input {bad:?} produced no warning at all"
            );
        }
        let result = import_d("   ", Fit::Bbox, 0.05, true);
        assert!(result.path.subpaths.is_empty());
    }
}
