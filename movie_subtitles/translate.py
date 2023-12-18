import logging

from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger("translate")


class Translate:
    def __init__(self, model_name: str = "jbochi/madlad400-3b-mt") -> None:
        self.model_name = model_name

        logger.info(f"Loading model {model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, device_map="auto"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)

    def __call__(self, text: str, output_lang: str) -> str:
        return self.translate(text, output_lang)

    def translate(self, text: str, output_lang: str) -> str:
        text = f"<2{output_lang}> {text}"
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.model.device
        )
        outputs = self.model.generate(input_ids=input_ids)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return text[1:] if text[0] == " " else text
