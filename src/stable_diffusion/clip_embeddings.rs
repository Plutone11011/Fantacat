use tokenizers::Tokenizer;
use candle_transformers::models::stable_diffusion;
use candle_core::Tensor;
use anyhow::{Error, Result};


use crate::stable_diffusion::stable_diffusion_files;

fn get_tokenizer() -> Result<Tokenizer>{

    let tokenizer_sd = stable_diffusion_files::StableDiffusionFiles::Tokenizer;

    let tokenizer_file = tokenizer_sd.get(None, true)?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(Error::msg)?;

    Ok(tokenizer)
    
}


