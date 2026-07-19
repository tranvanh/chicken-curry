mod tensor;
use tensor::Tensor;

fn main(){
    let tens = Tensor::new(
        vec![2,3,3],
        vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0, 12.0, 13.0,14.0,15.0,16.0,17.0,18.0],
    ).expect("Could not create tensor");
    println!("{}", tens);
    let value = tens.get(&[1,1,1]).unwrap();

    println!("at (1,1,1) = {}", value);
}
