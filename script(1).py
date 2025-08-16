# Now let me create the complete single-file emotion-aware chatbot application

chatbot_code = '''#!/usr/bin/env python3
"""
EMOTION-AWARE CHATBOT WITH CAMERA INTEGRATION
=============================================

A real-time emotion detection chatbot that uses computer vision to understand
human emotions and provides appropriate responses using quantized LLM models.

Author: AI Assistant
Requirements: opencv-python, deepface, transformers, torch, bitsandbytes, numpy, threading

Usage: python emotion_chatbot.py
"""

import cv2
import numpy as np
import torch
import threading
import time
import logging
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig,
        pipeline
    )
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.info("Install with: pip install opencv-python deepface transformers torch bitsandbytes accelerate")
    exit(1)


class EmotionDetector:
    """Handles real-time emotion detection from camera feed"""
    
    def __init__(self):
        self.emotion_history = deque(maxlen=10)  # Store last 10 emotions
        self.last_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion colors for display
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue  
            'angry': (0, 0, 255),      # Red
            'surprise': (255, 255, 0), # Cyan
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128),  # Dark Yellow
            'neutral': (255, 255, 255) # White
        }
        
        logger.info("EmotionDetector initialized successfully")
    
    def detect_emotion(self, frame: np.ndarray) -> Tuple[str, float, List]:
        """
        Detect emotion from a video frame
        Returns: (emotion, confidence, face_coordinates)
        """
        try:
            # Convert to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return self.last_emotion, self.emotion_confidence, []
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region with some padding
            padding = 20
            face_region = rgb_frame[max(0, y-padding):y+h+padding, 
                                  max(0, x-padding):x+w+padding]
            
            if face_region.size == 0:
                return self.last_emotion, self.emotion_confidence, [largest_face]
            
            # Analyze emotion using DeepFace
            result = DeepFace.analyze(face_region, 
                                    actions=['emotion'], 
                                    enforce_detection=False,
                                    silent=True)
            
            if isinstance(result, list):
                result = result[0]
            
            # Get dominant emotion
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion] / 100.0
            
            # Update emotion history
            self.emotion_history.append(dominant_emotion)
            
            # Smooth emotion detection (require consistency)
            if len(self.emotion_history) >= 3:
                recent_emotions = list(self.emotion_history)[-3:]
                if recent_emotions.count(dominant_emotion) >= 2:
                    self.last_emotion = dominant_emotion
                    self.emotion_confidence = confidence
            
            return self.last_emotion, self.emotion_confidence, [largest_face]
            
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return self.last_emotion, self.emotion_confidence, []
    
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """Get color for emotion display"""
        return self.emotion_colors.get(emotion, (255, 255, 255))


class QuantizedChatBot:
    """Chatbot using quantized LLM models"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self.max_history = 10
        
        # Emotion-based response templates
        self.emotion_prompts = {
            'happy': "The user seems very happy and joyful! Respond enthusiastically and share their positive energy.",
            'sad': "The user appears sad or down. Provide comfort, encouragement, and uplifting support.",
            'angry': "The user looks angry or frustrated. Use calm, soothing language to help them relax.",
            'surprise': "The user seems surprised! Show curiosity and engagement about what might have caused this.",
            'fear': "The user appears worried or afraid. Provide reassurance, comfort, and supportive words.",
            'disgust': "The user looks disgusted or uncomfortable. Gently change the topic to something more pleasant.",
            'neutral': "The user has a neutral expression. Maintain a friendly, conversational tone."
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load the quantized model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully with 4-bit quantization")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to CPU-only mode...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                logger.info("Model loaded in CPU mode")
            except Exception as e2:
                logger.error(f"Failed to load model in CPU mode: {e2}")
                raise
    
    def generate_emotion_response(self, emotion: str, user_input: str = "") -> str:
        """Generate response based on detected emotion"""
        try:
            # Create emotion-aware prompt
            emotion_context = self.emotion_prompts.get(emotion, self.emotion_prompts['neutral'])
            
            if user_input:
                prompt = f"{emotion_context} User said: '{user_input}'. Respond naturally and appropriately."
            else:
                prompt = f"{emotion_context} The user hasn't said anything yet, but you can see they look {emotion}. Say something appropriate to their emotional state."
            
            # Limit prompt length
            max_prompt_length = 200
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=100)
            
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = inputs.cuda()
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new generated part
            generated_text = response[len(prompt):].strip()
            
            # Clean up the response
            if not generated_text:
                generated_text = self._get_fallback_response(emotion)
            
            # Limit response length
            sentences = generated_text.split('.')
            if len(sentences) > 2:
                generated_text = '. '.join(sentences[:2]) + '.'
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(emotion)
    
    def _get_fallback_response(self, emotion: str) -> str:
        """Fallback responses when model fails"""
        fallback_responses = {
            'happy': "I can see you're happy! That's wonderful! 😊",
            'sad': "I notice you seem a bit down. I'm here if you want to talk. 💙",
            'angry': "You seem frustrated. Take a deep breath - everything will be okay. 🕊️",
            'surprise': "Oh! You look surprised! What's happening? 😲",
            'fear': "I can see you're worried. Remember, I'm here to help and support you. 🤗",
            'disgust': "Something bothering you? Let's talk about something more pleasant! 🌸",
            'neutral': "Hello! How are you feeling today? I'm here to chat! 👋"
        }
        return fallback_responses.get(emotion, "Hello! I'm here to chat with you! 😊")


class EmotionAwareChatBot:
    """Main application combining emotion detection and chatbot"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.emotion_detector = EmotionDetector()
        self.chatbot = QuantizedChatBot(model_name)
        self.cap = None
        self.running = False
        self.current_response = "Hello! I'm your emotion-aware AI companion! 🤖"
        self.last_emotion_time = time.time()
        self.emotion_response_delay = 3.0  # Seconds between emotion-based responses
        self.user_input = ""
        self.input_mode = False
        
        logger.info("EmotionAwareChatBot initialized")
    
    def start_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Could not open camera")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with emotion detection and chat display"""
        # Detect emotion
        emotion, confidence, faces = self.emotion_detector.detect_emotion(frame)
        
        # Draw face rectangles
        for (x, y, w, h) in faces:
            color = self.emotion_detector.get_emotion_color(emotion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display emotion and confidence
            emotion_text = f"{emotion.title()} ({confidence:.1%})"
            cv2.putText(frame, emotion_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Generate new response based on emotion change
        current_time = time.time()
        if (current_time - self.last_emotion_time) > self.emotion_response_delay:
            if emotion != getattr(self, 'last_processed_emotion', 'neutral'):
                self.last_processed_emotion = emotion
                self.last_emotion_time = current_time
                
                # Generate emotion-based response in background thread
                threading.Thread(
                    target=self._generate_background_response,
                    args=(emotion, self.user_input),
                    daemon=True
                ).start()
        
        # Display chat response
        self._draw_chat_interface(frame)
        
        return frame
    
    def _generate_background_response(self, emotion: str, user_input: str = ""):
        """Generate response in background thread"""
        try:
            new_response = self.chatbot.generate_emotion_response(emotion, user_input)
            self.current_response = new_response
            if user_input:
                self.user_input = ""  # Clear input after processing
        except Exception as e:
            logger.error(f"Background response generation failed: {e}")
    
    def _draw_chat_interface(self, frame: np.ndarray):
        """Draw chat interface on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for chat
        overlay = frame.copy()
        chat_height = 120
        cv2.rectangle(overlay, (0, height-chat_height), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Display current response (word wrapped)
        y_offset = height - chat_height + 20
        words = self.current_response.split(' ')
        line = ""
        
        for word in words:
            test_line = line + " " + word if line else word
            if len(test_line) > 70:  # Approximate character limit per line
                cv2.putText(frame, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                line = word
                y_offset += 20
                if y_offset > height - 10:  # Prevent text overflow
                    break
            else:
                line = test_line
        
        if line and y_offset <= height - 10:
            cv2.putText(frame, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display input prompt
        if self.input_mode:
            input_text = f"You: {self.user_input}_"
            cv2.putText(frame, input_text, (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Press 'i' to type, 'q' to quit", (10, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def handle_input(self, key: int):
        """Handle keyboard input"""
        if key == ord('i') and not self.input_mode:
            self.input_mode = True
            self.user_input = ""
        elif key == 13 and self.input_mode:  # Enter key
            if self.user_input.strip():
                # Process user input
                emotion = self.emotion_detector.last_emotion
                threading.Thread(
                    target=self._generate_background_response,
                    args=(emotion, self.user_input),
                    daemon=True
                ).start()
            self.input_mode = False
        elif key == 27 and self.input_mode:  # Escape key
            self.input_mode = False
            self.user_input = ""
        elif self.input_mode and key >= 32 and key < 127:  # Printable characters
            self.user_input += chr(key)
        elif self.input_mode and key == 8 and self.user_input:  # Backspace
            self.user_input = self.user_input[:-1]
    
    def run(self):
        """Main application loop"""
        if not self.start_camera():
            logger.error("Failed to start camera")
            return
        
        self.running = True
        logger.info("Starting emotion-aware chatbot...")
        logger.info("Controls: 'i' to type input, 'Enter' to send, 'Esc' to cancel input, 'q' to quit")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Emotion-Aware Chatbot', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                else:
                    self.handle_input(key)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


def main():
    """Main entry point"""
    print("=" * 60)
    print("EMOTION-AWARE CHATBOT")
    print("=" * 60)
    print("An AI companion that detects your emotions and responds accordingly!")
    print()
    print("Available Models (choose by number):")
    print("1. microsoft/DialoGPT-small (117M params) - Fast, conversation-focused")
    print("2. google/flan-t5-small (77M params) - Good instruction following")  
    print("3. HuggingFaceTB/SmolLM3-3B (3B params) - Best performance (requires more RAM)")
    print("4. meta-llama/Llama-3.2-1B (1B params) - Good balance")
    print()
    
    # Model selection
    model_choices = {
        "1": "microsoft/DialoGPT-small",
        "2": "google/flan-t5-small", 
        "3": "HuggingFaceTB/SmolLM3-3B",
        "4": "meta-llama/Llama-3.2-1B"
    }
    
    choice = input("Choose model (1-4, default=1): ").strip()
    model_name = model_choices.get(choice, "microsoft/DialoGPT-small")
    
    print(f"Selected model: {model_name}")
    print()
    print("Initializing chatbot...")
    
    try:
        # Create and run chatbot
        chatbot = EmotionAwareChatBot(model_name)
        chatbot.run()
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print("\\nTroubleshooting tips:")
        print("1. Ensure camera is connected and not used by other apps")
        print("2. Install dependencies: pip install opencv-python deepface transformers torch bitsandbytes")
        print("3. Ensure sufficient GPU memory for larger models")


if __name__ == "__main__":
    main()
'''

# Save the code to a file
with open("emotion_chatbot.py", "w", encoding="utf-8") as f:
    f.write(chatbot_code)

print("✅ Complete emotion-aware chatbot saved as 'emotion_chatbot.py'")
print(f"📄 File size: {len(chatbot_code):,} characters")
print("\n🔧 Key Features Implemented:")
print("• Real-time emotion detection using DeepFace + OpenCV")
print("• Quantized LLM integration (4-bit quantization)")
print("• Multiple model options under 6B parameters") 
print("• Emotion-aware response generation")
print("• Interactive chat interface")
print("• Background response generation for smooth UX")
print("• Comprehensive error handling and logging")
print("• Single-file implementation")