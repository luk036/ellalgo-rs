mod cutting_plane;
use crate::cutting_plane::cutting_plane_optim;

mod ell_calc;
use crate::ell_calc::EllCalc;

mod ell;
use crate::ell::Ell;

#[derive(Debug)]
enum CutChoices {
    Single(f64),
    Parallel(f64, Option<f64>),
}

trait IntoCutChoices {
    fn into(self) -> CutChoices;
}

impl CutChoices {
    fn new<A>(args: A) -> CutChoices
        where A: IntoCutChoices
    {
        args.into()
    }
}

impl IntoCutChoices for f64 {
    fn into(self) -> CutChoices {
        CutChoices::Single(self)
    }
}

impl IntoCutChoices for (f64, Option<f64>) {
    fn into(self) -> CutChoices {
        CutChoices::Parallel(self.0, self.1)
    }
}

fn main() {
    let x = CutChoices::new(2f64);
    let y = CutChoices::new((2f64, Some(2f64)));
    println!("{:#?}", x);
    println!("{:#?}", y);
}
