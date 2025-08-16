# Let me create a detailed implementation plan and write the single-file emotion-aware chatbot

# First, let me outline the structure and key components needed:

implementation_plan = """
EMOTION-AWARE CHATBOT - SINGLE FILE IMPLEMENTATION PLAN
=======================================================

1. DEPENDENCIES:
   - opencv-python (camera and face detection)
   - deepface (emotion recognition)
   - transformers (for Hugging Face LLMs)
   - torch (PyTorch for model inference)
   - bitsandbytes (for quantization)
   - pillow (image processing)
   - numpy (numerical operations)

2. KEY COMPONENTS:
   a) EmotionDetector class - handles camera input and emotion recognition
   b) ChatBot class - handles LLM interaction with quantized model
   c) EmotionAwareChatBot class - main application that integrates both
   d) UI/Display system - real-time camera feed with chat interface

3. WORKFLOW:
   - Initialize camera and emotion detection model
   - Load quantized LLM (SmolLM3-3B or DialoGPT-small)
   - Start video loop
   - For each frame:
     * Detect emotion from face
     * If emotion changes significantly, generate contextual response
     * Display emotion + chat response on video feed
   - Handle user text input for additional interaction

4. QUANTIZED MODEL OPTIONS (Under 6B params):
   - HuggingFaceTB/SmolLM3-3B (3B params) - best overall performance
   - microsoft/DialoGPT-small (117M params) - conversation focused
   - google/flan-t5-small (77M params) - instruction following
   - meta-llama/Llama-3.2-1B (1B params) - good balance

5. EMOTION-RESPONSE MAPPING:
   - Happy: Encourage and share positive energy
   - Sad: Provide comfort and uplifting messages
   - Angry: Use calming language and de-escalation
   - Surprised: Show curiosity and engagement
   - Neutral: Maintain normal conversation
   - Fear: Provide reassurance and support
   - Disgust: Change topic or provide gentle redirection
"""

print(implementation_plan)