# def main():
#     print("Hello from answer-ai-ocr!")


# if __name__ == "__main__":
#     main()
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# image = Image.open("test.png").convert("RGB")
# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)
# print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])