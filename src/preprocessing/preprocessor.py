import cv2
import numpy as np
from PIL import Image

class IrisPreprocessor:
    """Preprocess iris images for Vision Transformer"""
    
    @staticmethod
    def enhance_iris(image):
        """Enhance iris image quality"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5,5), 0)
        
        return blurred

class VeinPreprocessor:
    """Preprocess vein images for Vision Transformer"""
    
    @staticmethod
    def enhance_veins(image):
        """Enhance vein patterns"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Median filtering
        filtered = cv2.medianBlur(enhanced, 5)
        
        return filtered

class MultimodalPreprocessor:
    """Handle preprocessing of both iris and vein images"""
    
    def __init__(self):
        self.iris_preprocessor = IrisPreprocessor()
        self.vein_preprocessor = VeinPreprocessor()
    
    def preprocess_pair(self, iris_image, vein_image):
        """Preprocess iris-vein pair"""
        # Enhance images
        iris_enhanced = self.iris_preprocessor.enhance_iris(iris_image)
        vein_enhanced = self.vein_preprocessor.enhance_veins(vein_image)
        
        # Resize for ViT (224x224)
        iris_resized = cv2.resize(iris_enhanced, (224, 224))
        vein_resized = cv2.resize(vein_enhanced, (224, 224))
        
        # Convert to 3 channels
        iris_rgb = cv2.cvtColor(iris_resized, cv2.COLOR_GRAY2RGB)
        vein_rgb = cv2.cvtColor(vein_resized, cv2.COLOR_GRAY2RGB)
        
        return iris_rgb, vein_rgb
