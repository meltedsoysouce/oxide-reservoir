use crate::matrix_base::NodeMatrix;

/// The trait for forward propagation
pub trait Forward {
    fn forward(self: &mut Self, input: NodeMatrix) -> NodeMatrix;
}
