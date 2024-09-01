use ndarray::{Array1, Array2};

pub type NodeMatrix = Array1<f32>;
pub type WeightMatrix = Array2<f32>;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matrix_base() {
        let node_matrix = NodeMatrix::zeros(3);
        let weight_matrix = WeightMatrix::zeros((3, 3));
        assert_eq!(node_matrix, Array1::zeros(3));
        assert_eq!(weight_matrix, Array2::zeros((3, 3)));
    }

    #[test]
    fn test_matrix_shape() {
        let node = NodeMatrix::zeros(3);
        assert_eq!(node.shape(), &[3]);

        let weight = WeightMatrix::zeros((3, 3));
        assert_eq!(weight.shape(), &[3, 3]);
    }
}
