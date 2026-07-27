use chicken_curry::tensor::Tensor;

fn main() {
    let tens = Tensor::new(
        vec![3, 3, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        ],
    )
    .expect("Could not create tensor");
    // println!("{}", tens);
    //let value = tens.get(&[1,1,1,0]).unwrap();
    let a = 2.0 * &tens;
    let b = 3.0 * &tens;
    let mul = &a * &b;
    let printable = mul.relu().neg().abs();
    printable.print_computation_graph();
    // println!("{}", tens);
    // println!("------------");
    // println!("{}", test);
    // println!("#####");
    // println!("{}", &test + &tens);
}
