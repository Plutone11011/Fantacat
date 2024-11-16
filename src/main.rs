use clap::Parser;
use anyhow::Result;
use candle_transformers::models::stable_diffusion as sd;
use candle_core::Tensor;

// use hf_hub::api::tokio::Api;
// use candle_core::Device;
mod stable_diffusion;
mod image_utils;


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// prompt request for model
    #[arg(short='p', long="prompt", default_value_t=String::from("Red cat with white and red stripes (hd, realistic, high-def)."))]
    prompt: String,

    #[arg(long="uncond_prompt", default_value_t=String::from(""))]
    uncond_prompt: String,

    /// Number of images to generate
    #[arg(long="n_images", default_value_t = 1)]
    n_images: usize,

    /// Number of diffusion steps
    #[arg(long="n_steps", default_value_t = 5)]
    n_steps: usize,

    #[arg(long="intermediary_images", default_value_t = true)]
    intermediary_images: bool,

    #[arg(short='o', long="output")]
    final_image: String,

    #[arg(long="use_flash_attn", default_value_t = false)]
    use_flash_attn: bool,



}




fn run_diffusion(args: Args) -> Result<()> {
    

    let width = Some(640 as usize);
    let height = Some(480 as usize);
    let sd_config = sd::StableDiffusionConfig::v1_5(None, height, width);
    let n_steps = args.n_steps; 
    let device = &candle_core::Device::new_cuda(0)?;
    let scheduler = sd_config.build_scheduler(n_steps)?;
    let batch_size = 1;
    let dtype = candle_core::DType::F16;

    let prompt = args.prompt ;
    println!("Generate an image for prompt: {}", prompt);
    let vae_scale: f64 = 0.18215;
    // match sd_version {
    //     StableDiffusionVersion::V1_5
    //     | StableDiffusionVersion::V2_1
    //     | StableDiffusionVersion::Xl => 0.18215,
    //     StableDiffusionVersion::Turbo => 0.13025,
    // };

    let embeddings = {
        let tokenizer = stable_diffusion::clip_embeddings::get_tokenizer(None)?;
        let encoded_prompt = stable_diffusion::clip_embeddings::encode_prompt(&prompt, &tokenizer, &sd_config, device)?;
        let embedding_model = stable_diffusion::clip_embeddings::get_embedding_model(None, &sd_config, device)?;
        stable_diffusion::clip_embeddings::get_embeddings(&encoded_prompt, &embedding_model)
    }?;
    println!("Embeddings created.");
    let vae = stable_diffusion::vae::get_vae(None, &sd_config, device, dtype)?;
    println!("VAE created.");
    let unet = stable_diffusion::unet::get_unet(None, &sd_config, device, dtype, args.use_flash_attn)?;
    println!("UNet created");

    for idx in 0..args.n_images {
        println!("Generating image number {}", idx);
        let timesteps = scheduler.timesteps();
        println!("Scheduler timesteps instantiated");
        // randomly generate latent representation of image
        // TODO: img2img needs different approach
        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (batch_size, 4, sd_config.height / 8, sd_config.width / 8),
            &device,
        )?;

        // scale the initial noise by the standard deviation required by the scheduler
        latents = (latents * scheduler.init_noise_sigma())?;
        latents = latents.to_dtype(dtype)?;
        
        println!("Latents scaled by standard deviation");
        println!("Entering diffusion process. Iterating for {:?} timesteps", timesteps);

        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            
            let start_time = std::time::Instant::now();
            let latent_model_input = scheduler.scale_model_input(latents.clone(), timestep)?;

            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &embeddings)?;

            latents = scheduler.step(&noise_pred, timestep, &latents)?;

            let dt = start_time.elapsed().as_secs_f32();
            println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

            if args.intermediary_images {
                image_utils::save::save_batch_encoded_images(
                    &vae,
                    &latents,
                    vae_scale,
                    batch_size,
                    idx,
                    &args.final_image,
                    args.n_images,
                    Some(timestep_index + 1),
                )?;
            }
        }

        println!("Generating final image version for sample {}", idx);
        image_utils::save::save_batch_encoded_images(
            &vae,
            &latents,
            vae_scale,
            batch_size,
            idx,
            &args.final_image,
            args.n_images,
            None
        )?;

    }

    println!("Finished!");
    Ok(())
    // println!("{:?}", print_type_of(&weights))
}


fn main() -> Result<()>{
    let args = Args::parse();

    run_diffusion(args)
}