
use models::logisticregression::Model;
use std::error::Error;
use std::io;

mod models;
mod parsedata;

fn main() -> Result<(), Box<dyn Error>> {

    let (x0, y0, x1, y1)  = parsedata::penguinscsv(0.8)?;
    let mut model: Model = models::logisticregression::Model::new(x0.clone(), y0.clone());
    let weights: Vec<f64> = model.gradientdescent(0.1, 1000);

    let (cost, accuracy) = model.eval(&x1, &y1, &weights);

    println!("Weights: {:?}\nCost: {}\nAccuracy: {}%\n", weights, cost, accuracy);

    // inputting new penguin data from user
    // println!("INPUT: culmen_length_mm culmen_depth_mm flipper_length_mm body_mass_g");
    // let mut input = String::new();
    // io::stdin().read_line(&mut input).expect("ruh roh");

    // let pengdata: Vec<f64> = input
    //     .split_whitespace()
    //     .map(|s| s.parse().expect("nan"))
    //     .collect();

    // let container: Vec<Vec<f64>> = vec![pengdata];

    // let guess: Vec<u8> = model.predict(&container, &weights);

    // if guess[0] == 0 {
    //     println!("Adelie");
    // } 
    // if guess[0] == 1{
    //     println!("Gentoo");
    // }


    Ok(())
}
