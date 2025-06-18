#!/usr/bin/env python3
"""
FinGPT Forecaster Setup - DOW30 Llama2-7B LoRA
This script sets up the specific FinGPT forecaster model for stock prediction.
Based on: https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora
"""

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import sys
from huggingface_hub import login

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set your Hugging Face token
HUGGING_FACE_TOKEN = "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"


class FinGPTForecaster:
    def __init__(self):
        self.base_model = None
        self.tokenizer = None
        self.model = None
        self.device = self._get_best_device()

    def _get_best_device(self):
        """Determine the best device to use"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def authenticate_huggingface(self):
        """Login to Hugging Face using the provided token"""
        print("🔐 Authenticating with Hugging Face...")
        try:
            login(token=HUGGING_FACE_TOKEN)
            print("✅ Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            return False

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
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")

        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"Using device: {self.device}")

        print("✅ System check complete\n")

    def setup_model(self):
        """Set up the FinGPT forecaster model"""
        print("📥 Setting up FinGPT Forecaster (DOW30 Llama2-7B LoRA)...")

        try:
            # Load tokenizer for Llama2-7b-chat
            print("📥 Loading Llama2-7b-chat tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf"
            )

            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("✅ Tokenizer loaded successfully")

            # Load base model
            print("🔧 Loading Llama2-7b-chat base model...")
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            # Set device-specific parameters
            if self.device == "cuda":
                load_kwargs.update(
                    {
                        "device_map": "auto",
                        "torch_dtype": torch.float16,
                    }
                )
            elif self.device == "mps":
                load_kwargs.update(
                    {
                        "device_map": "mps",
                        "torch_dtype": torch.float16,
                    }
                )
            else:  # CPU
                load_kwargs.update(
                    {
                        "device_map": "cpu",
                        "torch_dtype": torch.float32,
                    }
                )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf", **load_kwargs
            )
            print("✅ Base model loaded successfully")

            # Load FinGPT forecaster adapter
            print("📥 Loading FinGPT forecaster adapter...")
            self.model = PeftModel.from_pretrained(
                self.base_model, "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
            )
            self.model = self.model.eval()
            print("✅ FinGPT forecaster adapter loaded successfully")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("\n💡 Common solutions:")
            print("1. Check your internet connection")
            print("2. Verify your Hugging Face token has access to the model")
            print("3. Ensure you have enough disk space and memory")
            return False

    def create_forecast_prompt(self, company_name, historical_data=None):
        """Create a proper forecast prompt for the model"""
        if historical_data:
            prompt = f"""Instruction: What is the market sentiment for {company_name} stock? Please provide analysis and forecast.

Historical Context: {historical_data}

Analysis:"""
        else:
            prompt = f"""Instruction: What is the market sentiment for {company_name} stock? Please provide analysis and forecast.

Analysis:"""

        return prompt

    def generate_forecast(self, company_name, historical_data=None, max_new_tokens=150):
        """Generate a stock forecast for the given company"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model not loaded. Please run setup first.")
            return None

        print(f"🔮 Generating forecast for {company_name}...")

        try:
            # Create the prompt
            prompt = self.create_forecast_prompt(company_name, historical_data)

            # Tokenize
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )

            # Move to device if not CPU
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            print("🤖 Generating prediction...")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    top_k=50,
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            forecast = response[len(prompt) :].strip()

            return {
                "company": company_name,
                "prompt": prompt,
                "forecast": forecast,
                "full_response": response,
            }

        except Exception as e:
            print(f"❌ Error during forecast generation: {str(e)}")
            return None

    def test_forecasting(self):
        """Test the model with sample DOW30 companies"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model not loaded. Please run setup first.")
            return

        print("\n🧪 Testing forecasting capabilities...")

        # Test with a few DOW30 companies
        test_companies = [
            "Apple Inc. (AAPL)",
            "Microsoft Corporation (MSFT)",
            "Amazon.com Inc. (AMZN)",
        ]

        for company in test_companies:
            print(f"\n{'='*60}")
            print(f"📊 Testing forecast for: {company}")
            print("=" * 60)

            result = self.generate_forecast(company)

            if result:
                print(f"\n🔮 Forecast for {result['company']}:")
                print("-" * 50)
                print(result["forecast"])
                print("-" * 50)
            else:
                print(f"❌ Failed to generate forecast for {company}")

            print("\n" + "=" * 60)

    def interactive_forecasting(self):
        """Interactive forecasting session"""
        if self.model is None or self.tokenizer is None:
            print("❌ Model not loaded. Please run setup first.")
            return

        print(f"\n💬 FinGPT Forecaster Interactive Session")
        print("🎯 Specialized for DOW30 stock forecasting")
        print("💡 Enter company names or stock symbols")
        print("Type 'quit' or 'exit' to end the session")
        print("-" * 60)

        while True:
            try:
                company = input("\n📈 Enter company name or symbol: ").strip()

                if company.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not company:
                    continue

                # Ask for historical data (optional)
                historical = input(
                    "📊 Enter historical context (optional, press Enter to skip): "
                ).strip()

                historical_data = historical if historical else None

                # Generate forecast
                result = self.generate_forecast(company, historical_data)

                if result:
                    print(f"\n🔮 Forecast for {result['company']}:")
                    print("=" * 60)
                    print(result["forecast"])
                    print("=" * 60)
                else:
                    print("❌ Failed to generate forecast. Please try again.")

            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("💡 Try again or type 'quit' to exit.")


def main():
    """Main setup and execution function"""
    print("🚀 FinGPT Forecaster Setup - DOW30 Llama2-7B LoRA")
    print("=" * 60)
    print("🎯 Specialized financial forecasting model")
    print(
        "📊 Based on: https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
    )
    print()

    forecaster = FinGPTForecaster()

    # Authenticate with Hugging Face
    if not forecaster.authenticate_huggingface():
        print("❌ Authentication failed. Cannot proceed without valid token.")
        return

    # Check system requirements
    forecaster.check_system_requirements()

    # Setup model
    print("🔧 Setting up FinGPT Forecaster...")
    if forecaster.setup_model():
        print("\n🎉 Model setup complete!")

        # Test forecasting
        forecaster.test_forecasting()

        # Ask user if they want interactive session
        while True:
            choice = (
                input("\n❓ Would you like to start interactive forecasting? (y/n): ")
                .strip()
                .lower()
            )
            if choice in ["y", "yes"]:
                forecaster.interactive_forecasting()
                break
            elif choice in ["n", "no"]:
                print("👍 Setup complete!")
                print("💡 Model is loaded and ready for forecasting.")
                break
            else:
                print("Please enter 'y' or 'n'")
    else:
        print("❌ Setup failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
