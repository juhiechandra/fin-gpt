#!/usr/bin/env python3
"""
FinGPT CPU-Only Working Setup for Apple Silicon Mac
This script uses CPU-only inference to avoid GPU memory and attention mechanism issues.
"""

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Force CPU usage to avoid MPS issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class FinGPTCPUWorking:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.model = None
        self.device = "cpu"  # Force CPU

    def check_system_requirements(self):
        """Check if system meets requirements"""
        print("🔍 Checking system requirements...")

        # Check Python version
        python_version = sys.version_info
        print(
            f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
        )

        # Check PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {self.device}")

        print("✅ System check complete\n")

    def download_and_setup_model(self):
        """Set up FinGPT using CPU-only approach"""
        print("📥 Loading Falcon-7B on CPU (this avoids GPU memory issues)...")
        print("⚠️  Note: CPU inference will be slower but more reliable")

        try:
            # Load tokenizer
            print("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")

            # Load model on CPU with safe settings
            print("🔧 Loading Falcon-7B model on CPU...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "tiiuae/falcon-7b",
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print("✅ Base model loaded successfully on CPU")

            # Try to load FinGPT adapter
            print("📥 Trying to load FinGPT Falcon adapter...")
            try:
                self.model = PeftModel.from_pretrained(
                    self.base_model, "FinGPT/fingpt-mt_falcon-7b_lora"
                )
                self.model = self.model.eval()
                print("✅ FinGPT Falcon adapter loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load FinGPT adapter: {str(e)}")
                print("📝 Using base Falcon-7B model for demonstration")
                self.model = self.base_model

            return True

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def test_inference(self):
        """Test the model with a simple financial query"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model not loaded. Please run setup first.")
            return

        print(f"\n🧪 Testing model inference on CPU...")

        # Simple financial prompt
        test_prompt = "Key factors for tech stock investment:"

        try:
            # Tokenize with conservative settings
            inputs = self.tokenizer(
                test_prompt, return_tensors="pt", truncation=True, max_length=100
            )

            print("🤖 Generating response (CPU inference may take 30-60 seconds)...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    top_p=0.9,
                    top_k=50,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_response = response[len(test_prompt) :].strip()

            print("\n📊 Model Response:")
            print("-" * 50)
            print(f"Question: {test_prompt}")
            print(f"Answer: {new_response}")
            print("-" * 50)
            print("✅ Test completed successfully!")

        except Exception as e:
            print(f"❌ Error during inference: {str(e)}")
            print("🔄 Trying with even simpler settings...")

            # Fallback with minimal settings
            try:
                simple_inputs = self.tokenizer(
                    "What is finance?", return_tensors="pt", max_length=20
                )

                with torch.no_grad():
                    simple_outputs = self.model.generate(
                        simple_inputs.input_ids,
                        max_new_tokens=10,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                simple_response = self.tokenizer.decode(
                    simple_outputs[0], skip_special_tokens=True
                )
                print(f"✅ Fallback test successful: {simple_response}")

            except Exception as fallback_e:
                print(f"❌ Fallback also failed: {str(fallback_e)}")

    def interactive_chat(self):
        """Simple interactive chat on CPU"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model not loaded. Please run setup first.")
            return

        print(f"\n💬 Starting CPU-based interactive chat...")
        print("⏰ Note: Responses will take 30-60 seconds on CPU")
        print("💡 Keep questions short and simple")
        print("Type 'quit' or 'exit' to end the session")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n👤 Your question: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                if len(user_input) > 50:
                    print(
                        "⚠️ Please keep questions under 50 characters for CPU processing."
                    )
                    continue

                # Simple tokenization
                inputs = self.tokenizer(
                    user_input, return_tensors="pt", truncation=True, max_length=30
                )

                print("🤖 Processing... (this will take a moment on CPU)")
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=20,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        top_p=0.9,
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                new_response = response[len(user_input) :].strip()

                print(f"\n🤖 Response: {new_response}")

            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("💡 Try a simpler question or type 'quit' to exit.")


def main():
    """Main setup and execution function"""
    print("🚀 FinGPT CPU-Only Working Setup")
    print("=" * 40)
    print("🔧 Reliable CPU-based inference for Apple Silicon")
    print("⚠️  Note: Slower but stable performance")
    print()

    setup = FinGPTCPUWorking()

    # Check system requirements
    setup.check_system_requirements()

    # Setup model
    print("🔧 Setting up CPU-only FinGPT...")
    if setup.download_and_setup_model():
        print("\n🎉 Model setup complete!")

        # Test inference
        setup.test_inference()

        # Ask user if they want interactive chat
        while True:
            choice = (
                input("\n❓ Would you like to start interactive chat? (y/n): ")
                .strip()
                .lower()
            )
            if choice in ["y", "yes"]:
                setup.interactive_chat()
                break
            elif choice in ["n", "no"]:
                print("👍 Setup complete!")
                print("💡 Model is loaded and ready for use.")
                break
            else:
                print("Please enter 'y' or 'n'")
    else:
        print("❌ Setup failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
