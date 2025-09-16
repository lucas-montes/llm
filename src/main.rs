use llm::{gemma3::{Gemma3, InitParams}, modules::Module};

fn main() {
    let mut gemma: Gemma3 = InitParams::gemma3_270m().into();
    let result = gemma.forward(&[1, 2, 3]);
    println!("{:?}", result.shape());
}
