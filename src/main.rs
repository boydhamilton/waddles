
use std::error::Error;

mod models;
mod parsedata;

fn main() -> Result<(), Box<dyn Error>> {

    let (x0, y0, x1, y1)  = parsedata::penguinscsv(0.3)?;

    // let mut model: Model = models::logisticregression::LModel::new(x0.clone(), y0.clone());
    // let weights: Vec<f64> = model.gradientdescent(0.1, 1000);

    // let (cost, accuracy) = model.eval(&x1, &y1, &weights);
    
    // println!("Weights: {:?}\nCost: {}\nAccuracy: {}%\n", weights, cost, accuracy);

    let mut model = models::projection::PModel::new(x0.clone(), y0.clone());

    model.weights();

    
    let mut score = 0;

    // zero is adelie, one is chinstrap, two is gentoo
    for i in 0..x1.len() {
        let guess = model.eval(&x1[i]);

        let mut anstest = 0.0;
        let mut bestlabel = 0;

        for i in 0..guess.len() {
            if guess[i] > anstest {
                anstest = guess[i];
                bestlabel = i;
            }
        }
        println!("Model answer: {}", bestlabel);
        println!("Real answer: {}", y1[i]);
        print!("\n");
        if bestlabel == y1[i] as usize {
            score += 1;
        }
    }
    
    let per = score as f64 / x1.len() as f64 * 100.0;

    println!("Test results, {}/{}\n {}", score, x1.len(), per);

    Ok(())
}
