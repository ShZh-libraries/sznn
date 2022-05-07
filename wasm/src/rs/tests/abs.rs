use sznn_wasm::*;
use sznn_wasm::layers::abs::handle_abs;
use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_abs_forward() {
    let input = Tensor::new(DTypes::I32(vec![1, -2, -3, 4]), vec![2, 2]);

    let output = handle_abs(&input);

    assert_eq!(output.get_length(), 4);
    assert_eq!(output.get_dim(), 2);
    assert_eq!(output.get_shape(), vec![2, 2]);
    assert_eq!(*output.get_data(), DTypes::I32(vec![1, 2, 3, 4]));
}
