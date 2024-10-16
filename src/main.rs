use clap::Parser;
use hf_hub::api::tokio::Api;
use candle_core::Device;
use std::any::type_name;
use tokio;

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>());
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// prompt request for model
    #[arg(short='p', long="prompt", default_value_t=String::from("Generate a red tabby cat."))]
    prompt: String,

    /// Number of images to generate
    #[arg(short='n', long="n_images", default_value_t = 1)]
    n_images: u8,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let api = Api::new().unwrap();
    let repo = api.model("stable-diffusion-v1-5/stable-diffusion-v1-5".to_string());
    let weights = repo.get("v1-5-pruned.safetensors").await.unwrap();
    let weights = candle_core::safetensors::load(weights, &Device::Cpu);

    println!("Model weights loaded!");
    println!("{:?}", print_type_of(&weights))
}
