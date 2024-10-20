use tokenizers::Tokenizer;
use candle_transformers::models::stable_diffusion;
use candle_core::Tensor;
use anyhow::{Error, Result};


use crate::stable_diffusion::stable_diffusion_files;

const CLIP_SPECIAL_TOKEN: &str = "<|endoftext|>";

fn get_tokenizer() -> Result<Tokenizer>{

    let tokenizer_sd = stable_diffusion_files::StableDiffusionFiles::Tokenizer;

    let tokenizer_file = tokenizer_sd.get(None, true)?;

    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(Error::msg)?;

    Ok(tokenizer)
    
}


pub fn get_padding_id(tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig) -> u32{
    // padding id depends on passed configuration
    let pad_id = match &stable_diffusion_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get(CLIP_SPECIAL_TOKEN).unwrap()
    };
    println!("Padding token id is {}", pad_id);
    pad_id
}


// pub fn get_embeddings(text :&str, tokenizer: &Tokenizer, stable_diffusion_config: &stable_diffusion::StableDiffusionConfig) -> Result<Tensor>{

//     let pad_id = get_padding_id(tokenizer, stable_diffusion_config);

//     println!("Embedding prompt {}", text);

//     let mut tokens = tokenizer
//         .encode(text, true)
//         .map_err(Error::msg)?
//         .get_ids()
//         .to_vec();

//     Ok(Tensor(tokens))

// }


// #[cfg(test)]
// mod tests {
//     use super::*;

//     let tokenizer ;

//     #[cfg(test)]
//     #[ctor::ctor]
//     fn init() {
//         tokenizer = get_tokenizer;
//     }

//     #[test]
//     fn get_padding_id() {