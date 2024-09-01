use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::{rand, RandomExt};

use crate::forward::Forward;
use crate::matrix_base::{NodeMatrix, WeightMatrix};

/// The input layer of echo state network
///
/// * 'weight' - The weight matrix of input layer
#[derive(Debug)]
pub struct Input {
    weight: WeightMatrix,
}

impl Input {
    /// Create a new input layer
    ///
    /// * 'input_dim' - The dimension of input
    /// * 'reservoir_size' - The size of reservoir
    /// * 'input_scale' - The scale of input
    /// * 'seed' - The seed of random number generator
    pub fn new(input_dim: usize, reservoir_size: usize, input_scale: f32, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Input {
            weight: WeightMatrix::random_using(
                (reservoir_size, input_dim),
                Uniform::new(-input_scale, input_scale),
                &mut rng,
            ),
        }
    }
}

impl Forward for Input {
    /// Forward propagation of input layer
    ///
    /// * 'input' - The input matrix(a raw data. The shape is (input_dim, batch_size))
    ///
    /// Return the output matrix of input layer.
    fn forward(self: &mut Self, input: NodeMatrix) -> NodeMatrix {
        self.weight.dot(&input)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_create_input() {
        let input = Input::new(3, 3, 1.0, 0);
        assert_eq!(input.weight.shape(), &[3, 3]);
    }

    #[test]
    fn test_forward_input() {
        let mut input = Input::new(3, 3, 1.0, 0);
        let input_data = NodeMatrix::from_vec(vec![1.0, 2.0, 3.0]);
        let output_matrix = input.forward(input_data);
        assert_eq!(output_matrix.shape(), &[3]);
    }
}
