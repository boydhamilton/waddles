

use std::f64::consts::E;


pub struct Model {
    datax: Vec<Vec<f64>>,
    datay: Vec<f64>
}

impl Model {

    pub fn new(inputx: Vec<Vec<f64>>, inputy: Vec<f64>) -> Self{
        Self{
            datax: inputx,
            datay: inputy
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + E.powf(-z))
    }

    // performance tracking
    // cost is low when predictions are accurate (probabilities close to actual labels)
    // vice versa for high cost
    pub fn eval(&self, x: &Vec<Vec<f64>>, y: &Vec<f64>, weights: &Vec<f64>) -> (f64, f64) {
        let m: f64 = y.len() as f64;

        let mut correct: f64 = 0.0;
        let mut cost: f64 = 0.0;

        for i in 0..y.len() {
            let mut z: f64 = 0.0;

            for j in 0..weights.len() { // dot product
                z += x[i][j] * weights[j];
            }
        
            let prediction: f64 = Self::sigmoid(z);

            println!("Prediction {} (Actual: {}): {}", i, y[i], prediction);
            if prediction.round() as u8 == y[i] as u8{
                correct += 1.0;
            }

            let epsilon: f64 = 0.0;//1e-12;
            cost += y[i] * (prediction + epsilon).ln() + (1.0 - y[i]) * (1.0 - prediction + epsilon).ln();
        }

        (-cost / m, correct * 100.0 / m )
    }

    pub fn gradientdescent(&mut self, learningrate: f64, iterations: usize) -> Vec<f64>{
        let m: f64 = self.datay.len() as f64;

        let mut weights: Vec<f64> = vec![0.0; self.datax[0].len()];

        let x: &Vec<Vec<f64>> = &self.datax;
        let y: &Vec<f64> = &self.datay;

        for _ in 0..iterations {
            let mut gradients: Vec<f64> = vec![0.0; weights.len()];

            for i in 0..x.len() {
                let mut z: f64 = 0.0;
                for j in 0..weights.len() { // dot product again should make a function
                    z += x[i][j] * weights[j]; 
                }

                let prediction: f64 = Self::sigmoid(z);

                for j in 0..weights.len() {
                    gradients[j] += (prediction - y[i]) * x[i][j];
                }
            }

            for j in 0..weights.len(){
                weights[j] -= learningrate * gradients[j] / m;
            }
        }

        weights
    }

    pub fn predict(&self, x: &Vec<Vec<f64>>, weights: &Vec<f64>) -> Vec<u8> {
        let mut predictions: Vec<u8> = Vec::new();

        for i in 0..x.len(){
            let mut z: f64 = 0.0;

            for j in 0..weights.len() {
                z += x[i][j] * weights[j];
            }

            let prediction: f64 = Self::sigmoid(z);

            if prediction >= 0.5 {
                predictions.push(1);
            } else{
                predictions.push(0);
            }
        }

        predictions
    }

}