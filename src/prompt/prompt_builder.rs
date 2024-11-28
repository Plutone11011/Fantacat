use std::string::ToString;

use super::prompt_entities::{Color, Medium, Style, Breed};


pub struct Prompt {
    medium: Option<Medium>,
    style: Option<Style>,    
    color: Option<Color>,
    breed: Option<Breed>,
    details: Option<String>
}


impl Prompt {
    pub fn builder() -> PromptBuilder {
        PromptBuilder::default()
    }

}

impl ToString for Prompt {
    fn to_string(&self) -> String {
        format!("{} {} {} cat {} {}", 
            self.style.as_ref().map_or_else(String::default, |s| s.to_string()), 
            self.color.as_ref().map_or_else(String::default, |s| s.to_string()),
            self.breed.as_ref().map_or_else(String::default, |s| s.to_string()), 
            self.medium.as_ref().map_or_else(String::default, |s| s.to_string()),
            self.details.as_ref().map_or_else(String::default, |s| s.to_string()))
    }
}

#[derive(Default)]
pub struct PromptBuilder {
    medium: Option<Medium>,
    style: Option<Style>,    
    color: Option<Color>,
    breed: Option<Breed>,
    details: Option<String>
}

impl PromptBuilder {

    pub fn set_medium(mut self, medium: Option<Medium>) -> Self{
        self.medium = medium;
        self
    }

    pub fn set_style(mut self, style: Option<Style>) -> Self{
        self.style = style;
        self
    }

    pub fn set_color(mut self, color: Option<Color>) -> Self{
        self.color = color;
        self
    }

    pub fn set_breed(mut self, breed: Option<Breed>) -> Self{
        self.breed = breed;
        self
    }

    pub fn set_details(mut self, details: Option<String>) -> Self{
        self.details = details;
        self
    }

    pub fn build(self) -> Prompt {
        Prompt {
            medium: self.medium,
            style: self.style,
            breed: self.breed,
            color: self.color,
            details: self.details
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn prompt_medium(){
        let builder = PromptBuilder::default();
        let test_medium = Medium::Photography;
        
        let result = builder.set_medium(Some(test_medium.clone()));
        
        assert_eq!(result.medium, Some(test_medium));
    }

    #[test]
    fn prompt_style(){
        let builder = PromptBuilder::default();
        let test_style = Style::Hyperrealist;
        
        let result = builder.set_style(Some(test_style.clone()));
        
        assert_eq!(result.style, Some(test_style));
    }

    #[test]
    fn prompt_set_color(){
        let builder = PromptBuilder::default();
        let test_color = Color::Silver;
        
        let result = builder.set_color(Some(test_color.clone()));
        
        assert_eq!(result.color, Some(test_color));
    }

    #[test]
    fn prompt_set_breed(){
        let builder = PromptBuilder::default();
        let test_breed = Breed::MaineCoon;

        let result = builder.set_breed(Some(test_breed.clone()));

        assert_eq!(result.breed, Some(test_breed));

        
    }

    #[test]
    fn prompt_set_details(){
        let prompt_builder = PromptBuilder::default();
        let details = prompt_builder.set_details(Some(String::from("stylish, sublime"))).details;

        assert_eq!("stylish, sublime", details.unwrap());
    }


    #[test]
    fn prompt_to_string(){
        let prompt_builder = PromptBuilder::default();

        let prompt = prompt_builder.set_style(None)
                                            .set_breed(Some(Breed::MaineCoon))
                                            .set_color(Some(Color::Red))
                                            .set_medium(Some(Medium::OilPainting))
                                            .set_details(Some(String::from("high quality")))
                                            .build();
        assert_eq!(" red maine-coon cat oil-painting high quality".to_string(), prompt.to_string());
    }
}