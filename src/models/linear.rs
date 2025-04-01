


pub struct Model {
    datax : Vec<f64>,
    datay : Vec<f64>,
    slope : f64
}

impl Model {

    pub fn new(inputx : Vec<f64>, inputy : Vec<f64>) -> Self {
        
        let slope = get_slope(&inputx, &inputy);

        Self {
            datax : inputx,
            datay : inputy,
            slope
        }
    }


    pub fn predict(&self, t : f64) -> f64 {
        self.slope * t + self.datay[0]
    }

}

fn get_slope(datax : &[f64], datay : &[f64]) -> f64{
    let n : f64 = datax.len() as f64;
    
    if n < 2.0 {
        return 0.0;
    }

    let mut sumx : f64 = 0.0;
    let mut sumy : f64 = 0.0;
    let mut sumxy : f64 = 0.0; 
    let mut sumx2 : f64 = 0.0;

    for i in 0..datax.len() as i32{
        let x : f64 = datax[i as usize];
        let y : f64 = datay[i as usize];
        sumx += x;
        sumy += y;
        sumxy +=  x* y;
        sumx2 += x * x;
    }

    return (n * sumxy - sumx * sumy) / (n * sumx2 - sumx * sumx);

}