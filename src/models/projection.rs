
// x and y are data to be trained upon, 
// pass data into eval function for the model to evaluate
pub struct PModel {
    x : Vec<Vec<f64>>,
    y : Vec<f64>,
    ideal : Vec<Vec<f64>>
}


// find "ideal" vector representing a given penguin, take unit vector (ideal vectors)
// take unit vector of given data, project onto each respective ideal vector
// find length of each projection, whichever one is longest is the choice

impl PModel {
    
    pub fn new(inputx : Vec<Vec<f64>>, inputy : Vec<f64>) -> Self{
        Self{
            x : inputx,
            y : inputy,
            ideal : Vec::new()
        }
    }

    pub fn eval( &self, x : &Vec<f64>) -> Vec<f64>{
        
        let mut proj_norms : Vec<f64> = Vec::new();

        let innorm = Self::norm(x);
        let unitinput = x.iter().map( |&x| x / innorm).collect::<Vec<_>>();

        for i in 0..self.ideal.len(){
            let proj : Vec<f64> = Self::proj(&unitinput, &self.ideal[i]);
            proj_norms.push(Self::norm(&proj));
            println!("{}", proj_norms[i]);
        }

        proj_norms
    }


    pub fn weights( &mut self ){
        // ith vector is vector one classes data (which is stored in vectors)
        let mut amassed : Vec<Vec<Vec<f64>>> = Vec::new(); 

        // get unit vectors
        for i in 0..self.x.len() as usize{

            let mut datavec: Vec<f64> = self.x[i].clone();

            let length: f64 = Self::norm(&datavec);

            datavec = datavec.iter().map( |&x| x / length).collect();

            let index  =self.y[i] as usize;
            
            if amassed.len() <= index {
                amassed.resize(index + 1, Vec::new());
            }

            amassed[index].push(datavec);
        }

        for i in 0..amassed.len() {
            let len =  amassed[i][amassed[i].len() - 1].len();
            let mut ideal_0 : Vec<f64> = vec![0.0; len];

            // sum them all, then divide all entries by the count to get average

            for j in 0..amassed[i].len() {
                for k in 0..amassed[i][j].len(){
                    ideal_0[k] += amassed[i][j][k];
                }
            }
            let sumquant : f64 = amassed[i].len() as f64;
            ideal_0 = ideal_0.iter().map( |&x| x / sumquant).collect();

            let norm = Self::norm(&ideal_0);
            ideal_0 = ideal_0.iter().map( |&x| x / norm).collect::<Vec<_>>();

            if self.ideal.len() <= i {
                self.ideal.resize(i + 1, Vec::new());
            }

            self.ideal[i] = ideal_0;

        }

        let mut c = 0;
        for idealv in self.ideal.clone() {
            print!("Generated class {} ideal vector <", c);
            for i in 0..idealv.len() {
                print!("{}, ", idealv[i]);
            }
            print!(">\n");
            c = c+1;
        }
    }

    // lin alg util

    fn norm( v : &Vec<f64>) -> f64{
        v.iter().map( |&x| x*x).sum::<f64>().sqrt()
    }

    // proj a onto v
    fn proj( a : &Vec<f64>, v : &Vec<f64>) -> Vec<f64>{
        //  = (a dot v) / len(v)^2 whole * v

        assert_eq!(a.len(), v.len()); // for dot product

        let mut dp: f64= 0.0;
        for i in 0..a.len(){
            dp += a[i] * v[i];
        }
        
        let projv_a_factor : f64 = dp / (Self::norm(v) * Self::norm(v));
        
        let projv_a: Vec<f64> = v.iter().map( |&x| x * projv_a_factor).collect();
        projv_a
    }

}