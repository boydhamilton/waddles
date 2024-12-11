
use models::logisticregression::Model;
use std::error::Error;
use std::io;

mod models;
mod parsedata;

fn main() -> Result<(), Box<dyn Error>> {

    let (x, y)  = parsedata::penguinscsv()?;
    let mut model: Model = models::logisticregression::Model::new(x, y);
    let weights: Vec<f64> = model.gradientdescent(0.1, 10000);

    println!("Weights: {:?}\n", weights);

    // inputting new penguin data from user
    println!("INPUT: culmen_length_mm culmen_depth_mm flipper_length_mm body_mass_g");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("ruh roh");

    let pengdata: Vec<f64> = input
        .split_whitespace()
        .map(|s| s.parse().expect("nan"))
        .collect();

    let container: Vec<Vec<f64>> = vec![pengdata];

    let guess: Vec<u8> = model.predict(&container, &weights);
    println!("{:?}", guess);

    Ok(())
}

