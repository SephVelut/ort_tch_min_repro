use ort::{ep, session::Session, value::Tensor as OrtTensor};
use tch::Device;

const ORT_LIB: &str = "onnxruntime-linux-x64-gpu-1.24.4/lib/libonnxruntime.so.1.24.4";
const MODEL: &str = "decoder_joint-model.onnx";

fn main() {
    let _ = Device::Cuda(0);

    ort::init_from(ORT_LIB).unwrap().commit();

    let mut session = Session::builder()
        .unwrap()
        .with_execution_providers([ep::CUDA::default().with_device_id(0).build().error_on_failure()])
        .unwrap()
        .commit_from_file(MODEL)
        .unwrap();

    let _ = session
        .run(ort::inputs! {
            "encoder_outputs" => OrtTensor::from_array(([1_usize, 1024_usize, 1_usize], vec![0.0_f32; 1024])).unwrap(),
            "targets" => OrtTensor::from_array(([1_usize, 1_usize], vec![0_i32; 1])).unwrap(),
            "target_length" => OrtTensor::from_array(([1_usize], vec![1_i32])).unwrap(),
            "input_states_1" => OrtTensor::from_array(([2_usize, 1_usize, 640_usize], vec![0.0_f32; 2 * 640])).unwrap(),
            "input_states_2" => OrtTensor::from_array(([2_usize, 1_usize, 640_usize], vec![0.0_f32; 2 * 640])).unwrap(),
        })
        .unwrap();
}
