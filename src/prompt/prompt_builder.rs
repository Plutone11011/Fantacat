use std::string::ToString;


struct Prompt {
    medium: Option<String>,
    style: Option<String>,    
    color: Option<String>,
    details: Option<String>
}


impl Prompt {
    pub fn builder() -> PromptBuilder {
        PromptBuilder::default()
    }

}

impl ToString for Prompt {
    fn to_string(&self) -> String {
        format!("{} {} cat {} {}", 
            self.style.as_ref().map_or_else(String::default, |s| s.to_string()), 
            self.color.as_ref().map_or_else(String::default, |s| s.to_string()), 
            self.medium.as_ref().map_or_else(String::default, |s| s.to_string()),
            self.details.as_ref().map_or_else(String::default, |s| s.to_string()))
    }
}

#[derive(Default)]
pub struct PromptBuilder {
    medium: Option<String>,
    style: Option<String>,    
    color: Option<String>,
    details: Option<String>
}

impl PromptBuilder {

    pub fn new() -> PromptBuilder{
        PromptBuilder {
            medium: None,
            style: None,
            color: None,
            details: None
        }
    }

    pub fn set_medium(mut self, medium: &str) -> Self{
        self.medium = Some(medium.to_string());
        self
    }

    pub fn set_style(mut self, style: &str) -> Self{
        self.style = Some(style.to_string());
        self
    }

    pub fn set_color(mut self, color: &str) -> Self{
        self.color = Some(color.to_string());
        self
    }

    pub fn set_details(mut self, details: &str) -> Self{
        self.details = Some(details.to_string());
        self
    }

    pub fn build(self) -> Prompt {
        Prompt {
            medium: self.medium,
            style: self.style,
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
        let prompt_builder = PromptBuilder::new();
        let medium = prompt_builder.set_medium("photograph").medium;

        assert_eq!("photograph", medium.unwrap());
    }

    #[test]
    fn prompt_style(){
        let prompt_builder = PromptBuilder::new();
        let style = prompt_builder.set_style("pixel-art").style;

        assert_eq!("pixel-art", style.unwrap());
    }

    #[test]
    fn prompt_set_color(){
        let prompt_builder = PromptBuilder::new();
        let color = prompt_builder.set_color("silver").color;

        assert_eq!("silver", color.unwrap());
    }

    #[test]
    fn prompt_set_details(){
        let prompt_builder = PromptBuilder::new();
        let details = prompt_builder.set_details("stylish, sublime").details;

        assert_eq!("stylish, sublime", details.unwrap());
    }


    #[test]
    fn prompt_build(){

        let prompt_builder = PromptBuilder::new();

        let prompt = prompt_builder.set_style("dark fantasy").set_color("silver").set_medium("oil painting").set_details("high quality").build();

        assert_eq!(prompt.medium.unwrap(), "oil painting");
        assert_eq!(prompt.style.unwrap(), "dark fantasy");
        assert_eq!(prompt.color.unwrap(), "silver");
        assert_eq!(prompt.details.unwrap(), "high quality");
    }

    #[test]
    fn prompt_to_string(){
        let prompt_builder = PromptBuilder::new();

        let prompt = prompt_builder.set_style("dark fantasy").set_color("silver").set_medium("oil painting").set_details("high quality").build();

        assert_eq!("dark fantasy silver cat oil painting high quality".to_string(), prompt.to_string());
    }
}