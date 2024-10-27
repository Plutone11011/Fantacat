use candle_transformers::models::stable_diffusion as sd;
use candle_core::{DType, Device, IndexOp, Tensor};
use anyhow::Result;

fn output_filename(
    basename: &str,
    sample_idx: usize,
    num_samples: usize,
    timestep_idx: Option<usize>,
) -> String {
    let filename = if num_samples > 1 {
        match basename.rsplit_once('.') {
            None => format!("{basename}.{sample_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}.{sample_idx}.{extension}")
            }
        }
    } else {
        basename.to_string()
    };
    match timestep_idx {
        None => filename,
        Some(timestep_idx) => match filename.rsplit_once('.') {
            None => format!("{filename}-{timestep_idx}.png"),
            Some((filename_no_extension, extension)) => {
                format!("{filename_no_extension}-{timestep_idx}.{extension}")
            }
        },
    }
}

/// Saves an image to disk using the image crate, this expects an input with shape
/// (c, height, width).
pub fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let p = p.as_ref();
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        panic!("save_image expects an input of shape (3, height, width)")
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => panic!("error saving image {p:?}"),
        };
    image.save(p).map_err(anyhow::Error::msg)?;
    Ok(())
}


pub fn save_batch_encoded_images(
    vae: &sd::vae::AutoEncoderKL,
    latents: &candle_core::Tensor,
    vae_scale: f64,
    batch_size: usize,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> Result<()> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..batch_size {
        let image = images.i(batch)?;
        let image_filename = output_filename(
            final_image,
            (batch_size * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        println!("Save image in {}", image_filename);
        save_image(&image, image_filename)?;
    }
    Ok(())
}
