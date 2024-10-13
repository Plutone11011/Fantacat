use clap::Parser;

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

fn main() {
    let args = Args::parse();



    println!("Hello, world!");
}
