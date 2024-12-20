
use csv::{Reader, ReaderBuilder, StringRecord};
use rand::rngs::ThreadRng;
use std::fs::File;
use std::error::Error;
use rand::seq::SliceRandom;


// TODO: for peng, take equal number from both species, random doesnt guarentee both will be trained equally
fn shuffled(x: &mut Vec<Vec<f64>>, y: &mut Vec<f64>, rng: &mut rand::rngs::ThreadRng) {
    // python esque line of code
    let mut combined: Vec<(Vec<f64>, f64)> = x.iter().cloned().zip(y.iter().cloned()).collect();
    
    combined.shuffle(rng);

    *x = combined.iter().map(|(features, _)| features.clone()).collect();
    *y = combined.iter().map(|(_, label)| *label).collect();
}

fn splitd(x: &Vec<Vec<f64>>, y: &Vec<f64>, train_size: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {

    let total_samples = x.len();
    let train_count = (train_size * total_samples as f64).round() as usize;

    let (x0, x1) = x.split_at(train_count);
    let (y0, y1) = y.split_at(train_count);

    (x0.to_vec(), y0.to_vec(), x1.to_vec(), y1.to_vec())
}

// return xtrain ytrain xtest ytest
pub fn penguinscsv(trainp : f64) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {

    // use penguins cut as we just classify between two (adelie and gentoo)
    // split at line 154, can look at splitting into train and test
    // todo: classify all three
    let csvpath: &str = "src/data/penguinscut.csv";

    let mut rdr: Reader<File> = ReaderBuilder::new().has_headers(true).from_path(csvpath)?;

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record:StringRecord = result?;
        let label: f64 = match record.get(0) {
            Some("Adelie") => 0.0,
            Some("Gentoo") => 1.0,
            _ => continue,
        };
        let features: Vec<f64> = (2..5)
            .filter_map(|i| record.get(i).and_then(|v| v.parse::<f64>().ok()))
            .collect();

        if features.len() == 3 { 
            y.push(label);
            x.push(features);
        }
    }

    let mut rng: ThreadRng = rand::thread_rng();
    shuffled(&mut x, &mut y, &mut rng);

    Ok(splitd(&x, &y, trainp))

}