
use clap::Parser;
use anyhow::Result;
use candle_transformers::models::stable_diffusion as sd;
use candle_core::{Tensor};
use stable_diffusion::stable_diffusion_files;


// use hf_hub::api::tokio::Api;
// use candle_core::Device;
mod stable_diffusion;
mod image_utils;
mod prompt;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// prompt request for model
    

    /// Number of images to generate
    #[arg(long="n_images", default_value_t = 1)]
    n_images: usize,

    /// Number of diffusion steps
    #[arg(long="n_steps", default_value_t = 5)]
    n_steps: usize,

    #[arg(long="width", default_value_t = 480)]
    width: usize,

    #[arg(long="height", default_value_t = 480)]
    height: usize,

    #[arg(long="intermediary_images", default_value_t = true)]
    intermediary_images: bool,

    #[arg(short='o', long="output")]
    final_image: String,

    #[arg(long="use_flash_attn", default_value_t = false)]
    use_flash_attn: bool,

    #[arg(short='g', long="guidance_scale")]
    guidance_scale: Option<f64>,
    
    #[arg(long, value_enum, default_value = "v2_1")]
    sd_version: stable_diffusion::stable_diffusion_files::StableDiffusionVersion,

    #[arg(long="medium")]
    medium: Option<prompt::prompt_entities::Medium>,

    #[arg(long="breed")]
    breed: Option<prompt::prompt_entities::Breed>,

    #[arg(long="style")]
    style: Option<prompt::prompt_entities::Style>,

    #[arg(long="color")]
    color: Option<prompt::prompt_entities::Color>,

    #[arg(long="details")]
    details: Option<String>,

}




fn run_diffusion(args: Args) -> Result<()> {
    

    let width = Some(args.width);
    let height = Some(args.height);
    let sd_version = args.sd_version;
    let sd_config = stable_diffusion::stable_diffusion_files::get_sd_config_from_version(&sd_version, None, height, width);
    let n_steps = args.n_steps; 
    let device = &candle_core::Device::new_cuda(0)?;
    // let device = &candle_core::Device::Cpu;
    let scheduler = sd_config.build_scheduler(n_steps)?;
    let batch_size = 1;
    let dtype = candle_core::DType::F16;
    let guidance_scale = match args.guidance_scale {
        None => 7.5,
        Some(guidance_scale) => guidance_scale
    };
    let use_guidance_scale = guidance_scale > 1.0;
    let t_start = 0 ; // relevant for img2img
    // let guidance_scale = match guidance_scale {
    //     Some(guidance_scale) => guidance_scale,
    //     None => match sd_version {
    //         StableDiffusionVersion::V1_5
    //         | StableDiffusionVersion::V2_1
    //         | StableDiffusionVersion::Xl => 7.5,
    //         StableDiffusionVersion::Turbo => 0.,
    //     },
    // };
    let color = args.color;
    let style = args.style;
    let medium = args.medium;
    let breed = args.breed;
    let details = args.details;
    let prompt_builder = prompt::prompt_builder::PromptBuilder::default();
    let prompt = prompt_builder.set_breed(breed)
                                        .set_color(color)
                                        .set_details(details)
                                        .set_medium(medium)
                                        .set_style(style)
                                        .build();

    let prompt = prompt.to_string();
    let uncond_prompt = "" ;
    println!("Generate an image for prompt: {}", prompt);
    let vae_scale: f64 = stable_diffusion::vae::get_vae_scale(&sd_version);

    
    

    let embeddings = {
        let tokenizer = stable_diffusion::clip_embeddings::get_tokenizer(None, &sd_version)?;
        let encoded_prompt = stable_diffusion::clip_embeddings::encode_prompt(&prompt, &tokenizer, &sd_config, device)?;
        let embedding_model = stable_diffusion::clip_embeddings::get_embedding_model(None, &sd_config, &sd_version, device)?;
        if use_guidance_scale {
            let encoded_uncond_prompt = stable_diffusion::clip_embeddings::encode_prompt(&uncond_prompt, &tokenizer, &sd_config, device)?;
            stable_diffusion::clip_embeddings::get_embeddings_for_guidance_scale(&encoded_prompt, &encoded_uncond_prompt,&embedding_model)
        }
        else {
            stable_diffusion::clip_embeddings::get_embeddings(&encoded_prompt, &embedding_model)
        }
        
    }?;
    println!("Embeddings created {:?}.", embeddings.shape());
    // only needed for turbo and xl since they use different embedding models
    // let text_embeddings = Tensor::cat(&embeddings, D::Minus1)?;
    let embeddings = embeddings.repeat((batch_size, 1, 1))?;
    println!("Batch of embeddings created {:?}.", embeddings.shape());

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
            
            if timestep_index < t_start {
                continue;
            }
            let start_time = std::time::Instant::now();

            let latent_model_input = if use_guidance_scale {
                // with guidance scale, need to start from duplicated latents
                // because model will process prompt and unconditional prompt simultaneously
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;

            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &embeddings)?;
            
            let noise_pred = if use_guidance_scale {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
            } else {
                noise_pred
            };

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