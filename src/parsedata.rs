
use csv::{Reader, ReaderBuilder, StringRecord};
use std::fs::File;
use std::error::Error;


pub fn penguinscsv() -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {

    // use penguins cut as we just classify between two (adelie and gentoo)
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
    
    Ok((x, y))

}