
// https://dafriedman97.github.io/mlbook/content/c7/concept.html

// i need the penguins. find a way to do data parsing
// https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris

mod models;

fn main() {

    let x = vec![
        vec![1.0, 1.0, 2.0],
        vec![1.0, 2.0, 3.0],
        vec![1.0, 3.0, 4.0],
        vec![1.0, 4.0, 5.0],
        vec![1.0, 5.0, 6.0],
    ];
    let y = vec![0.0, 0.0, 1.0, 1.0, 1.0]; // labels

    let mut model = models::logisticregression::Model::new(x[0..3].to_vec(), y[0..3].to_vec());

    // train using gd
    let trained_weights = model.gradientdescent(0.1, 1000);

    println!("Trained weights: {:?}", trained_weights);


    let predictions = model.predict(&x[3..].to_vec(), &trained_weights);
    println!("Predictions: {:?}", predictions);
}