use csv::{ReaderBuilder, StringRecord};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use std::error::Error;

// Shuffle the dataset (x and y) together randomly.
fn shuffled(x: &mut Vec<Vec<f64>>, y: &mut Vec<f64>, rng: &mut ThreadRng) {
    let mut combined: Vec<(Vec<f64>, f64)> = x.iter().cloned().zip(y.iter().cloned()).collect();
    combined.shuffle(rng);
    *x = combined.iter().map(|(features, _)| features.clone()).collect();
    *y = combined.iter().map(|(_, label)| *label).collect();
}

// Function to load data and split it into training and testing sets.
pub fn penguinscsv(trainp: f64) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let csvpath: &str = "src/data/penguins.csv";
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(csvpath)?;

    // Separate data by species.
    let mut adelie_x: Vec<Vec<f64>> = Vec::new();
    let mut adelie_y: Vec<f64> = Vec::new();
    let mut chinstrap_x: Vec<Vec<f64>> = Vec::new();
    let mut chinstrap_y: Vec<f64> = Vec::new();
    let mut gentoo_x: Vec<Vec<f64>> = Vec::new();
    let mut gentoo_y: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record: StringRecord = result?;
        let label: f64 = match record.get(0) {
            Some("Adelie Penguin (Pygoscelis adeliae)") => 0.0,
            Some("Chinstrap penguin (Pygoscelis antarctica)") => 1.0,
            Some("Gentoo penguin (Pygoscelis papua)") => 2.0,
            _ => continue,
        };

        // Parse features from columns 2, 3, and 4.
        let features: Vec<f64> = (2..5)
            .filter_map(|i| record.get(i).and_then(|v| v.parse::<f64>().ok()))
            .collect();

        if features.len() == 3 {
            match label {
                0.0 => { adelie_x.push(features.clone()); adelie_y.push(label); }
                1.0 => { chinstrap_x.push(features.clone()); chinstrap_y.push(label); }
                2.0 => { gentoo_x.push(features.clone()); gentoo_y.push(label); }
                _ => {}
            }
        }
    }

    // Calculate train size per species.
    let adelie_train_count = (adelie_x.len() as f64 * trainp).round() as usize;
    let chinstrap_train_count = (chinstrap_x.len() as f64 * trainp).round() as usize;
    let gentoo_train_count = (gentoo_x.len() as f64 * trainp).round() as usize;

    // Split each species' data into train/test sets.
    let (adelie_x_train, adelie_x_test) = adelie_x.split_at(adelie_train_count);
    let (adelie_y_train, adelie_y_test) = adelie_y.split_at(adelie_train_count);

    let (chinstrap_x_train, chinstrap_x_test) = chinstrap_x.split_at(chinstrap_train_count);
    let (chinstrap_y_train, chinstrap_y_test) = chinstrap_y.split_at(chinstrap_train_count);

    let (gentoo_x_train, gentoo_x_test) = gentoo_x.split_at(gentoo_train_count);
    let (gentoo_y_train, gentoo_y_test) = gentoo_y.split_at(gentoo_train_count);

    // Combine train/test sets for all species.
    let mut x_train = [adelie_x_train, chinstrap_x_train, gentoo_x_train].concat();
    let mut y_train = [adelie_y_train, chinstrap_y_train, gentoo_y_train].concat();
    let mut x_test = [adelie_x_test, chinstrap_x_test, gentoo_x_test].concat();
    let mut y_test = [adelie_y_test, chinstrap_y_test, gentoo_y_test].concat();

    // Shuffle the train and test datasets.
    let mut rng: ThreadRng = rand::thread_rng();
    shuffled(&mut x_train, &mut y_train, &mut rng);
    shuffled(&mut x_test, &mut y_test, &mut rng);

    Ok((x_train, y_train, x_test, y_test))
}
