import easyocr
import numpy as np
import torch
from typing import Union, List, Tuple, Dict, Any
from src.features.clip_extractor import CLIPExtractor

class OCRExtractor:
    """
    OCR Module for text extraction from images as part of the EVIS system.
    Uses EasyOCR for detection and recognition, and CLIP for text embedding.
    """
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True, clip_extractor: CLIPExtractor = None):
        """
        Initialize the EasyOCR reader and optionally a CLIP extractor for embeddings.
        
        Args:
            languages: List of language codes to support (e.g., ['en']).
            gpu: Whether to use GPU for OCR (requires CUDA).
            clip_extractor: An instance of CLIPExtractor. If None, one will be created when needed.
        """
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.clip_extractor = clip_extractor
        
    def extract_text(self, image: Union[str, np.ndarray]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract text from an image.
        
        Args:
            image: Path to the image file or a numpy array (OpenCV/PIL format).
            
        Returns:
            A tuple containing:
            - concatenated_text: All detected text joined by spaces.
            - detailed_results: List of dicts with 'text', 'confidence', and 'bbox'.
        """
        # EasyOCR readtext returns: (bbox, text, confidence)
        results = self.reader.readtext(image)
        
        detailed_results = []
        texts = []
        
        for (bbox, text, prob) in results:
            texts.append(text)
            detailed_results.append({
                'text': text,
                'confidence': float(prob),
                'bbox': [list(map(float, pt)) for pt in bbox]
            })
            
        concatenated_text = " ".join(texts)
        return concatenated_text, detailed_results

    def get_ocr_embedding(self, text: str) -> torch.Tensor:
        """
        Get an embedding for the OCR text using CLIP's text encoder.
        This ensures the OCR text is in the same semantic space as images and other text.
        
        Args:
            text: The text to embed.
            
        Returns:
            A tensor of shape (1, 512) containing the normalized embedding.
        """
        if not text:
            # Return a zero tensor if text is empty
            return torch.zeros((1, 512))
            
        if self.clip_extractor is None:
            self.clip_extractor = CLIPExtractor()
            
        return self.clip_extractor.extract_text_features(text)

if __name__ == "__main__":
    # Test block
    print("Testing OCRExtractor...")
    try:
        # We use a placeholder image check or a dummy array
        extractor = OCRExtractor(gpu=torch.cuda.is_available())
        
        # Create a blank image with some text would be hard without cv2/PIL here, 
        # so we'll just demonstrate the call structure.
        # In a real scenario, you'd pass a path to an image.
        
        test_text = "Hello EVIS System"
        print(f"Testing embedding for: '{test_text}'")
        embedding = extractor.get_ocr_embedding(test_text)
        print(f"Embedding shape: {embedding.shape}")
        
        print("\nOCR Extraction requires an actual image. To test with an image:")
        print("results = extractor.extract_text('path/to/image.jpg')")
        print("print(results[0]) # Concatenated text")
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")