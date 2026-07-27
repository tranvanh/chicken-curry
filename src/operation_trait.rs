#[allow(dead_code)]
pub(crate) trait Unary{
    fn map<F>(&self, f: F) -> Self where F: Fn(f32) -> f32;
    fn abs(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn ln(&self) -> Self;
    fn relu(&self) -> Self;
    fn neg(&self) -> Self;

    fn exp(&self) -> Self;

    fn pow(&self, n: i32) -> Self;
    fn powf(&self, n: f32) -> Self;
}
