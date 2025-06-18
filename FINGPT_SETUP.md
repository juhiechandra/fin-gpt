# FinGPT Forecaster Setup Guide

This guide will help you set up and run the **FinGPT Forecaster** model specifically designed for DOW30 stock predictions.

## 🎯 Model Information

- **Model**: [FinGPT/fingpt-forecaster_dow30_llama2-7b_lora](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)
- **Base Model**: Llama2-7b-chat-hf
- **Specialization**: DOW30 stock forecasting
- **Type**: LoRA (Low-Rank Adaptation) fine-tuned model

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
./setup_and_run.sh
```

This script will:

- Create a virtual environment
- Install all dependencies
- Set up your Hugging Face token
- Run the FinGPT forecaster

### Option 2: Manual Setup

1. **Create and activate virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Test the setup:**

   ```bash
   python3 test_fingpt.py
   ```

4. **Run the forecaster:**

   ```bash
   python3 fingpt_forecaster_setup.py
   ```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB recommended)
- **Storage**: ~15GB free space for model downloads
- **Internet**: Required for initial model download
- **Hugging Face Token**: Already configured in the scripts

## 🔧 System Requirements

### Recommended Hardware

- **CPU**: Modern multi-core processor
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, will use CPU if unavailable)
- **Apple Silicon**: Full MPS support for M1/M2/M3 Macs

### Supported Devices

- **CUDA**: NVIDIA GPUs (automatic detection)
- **MPS**: Apple Silicon Macs (automatic detection)
- **CPU**: Fallback for all systems (slower but works)

## 📊 Features

### Core Capabilities

- **Stock Forecasting**: Specialized predictions for DOW30 companies
- **Market Sentiment Analysis**: Analyzes market conditions
- **Interactive Chat**: Real-time forecasting sessions
- **Batch Processing**: Test multiple stocks at once

### Supported Companies

The model is specifically trained on DOW30 companies including:

- Apple Inc. (AAPL)
- Microsoft Corporation (MSFT)
- Amazon.com Inc. (AMZN)
- Tesla Inc. (TSLA)
- And all other DOW30 constituents

## 🎮 Usage Examples

### Basic Forecasting

```python
from fingpt_forecaster_setup import FinGPTForecaster

# Initialize the forecaster
forecaster = FinGPTForecaster()
forecaster.authenticate_huggingface()
forecaster.setup_model()

# Generate a forecast
result = forecaster.generate_forecast("Apple Inc. (AAPL)")
print(result['forecast'])
```

### Interactive Session

Run the main script and choose interactive mode to get real-time forecasts for any company.

### With Historical Context

```python
historical_data = "Stock has been trending upward for the past week due to strong earnings report"
result = forecaster.generate_forecast("Microsoft Corporation (MSFT)", historical_data)
```

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Error:**
   - Verify your Hugging Face token is correct
   - Check internet connection
   - Ensure token has access to the model

2. **Memory Issues:**
   - Close other applications
   - Use CPU mode if GPU memory is insufficient
   - Reduce `max_new_tokens` parameter

3. **Slow Performance:**
   - CPU inference is slower but reliable
   - Consider using a GPU if available
   - Keep prompts concise for faster processing

4. **Import Errors:**

   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Performance Tips

- **GPU Users**: The model will automatically use CUDA if available
- **Mac Users**: MPS acceleration is automatically enabled on Apple Silicon
- **CPU Users**: Expect 30-60 seconds per prediction
- **Memory Optimization**: Close unused applications for better performance

## 📁 File Structure

```
fin-gpt/
├── fingpt_forecaster_setup.py    # Main FinGPT forecaster implementation
├── test_fingpt.py                # Dependency testing script
├── setup_and_run.sh             # Automated setup script
├── requirements.txt              # Python dependencies
├── FINGPT_SETUP.md              # This guide
└── fingpt_cpu_working.py         # Previous Falcon-based implementation
```

## 🔐 Security Notes

- Your Hugging Face token is included in the scripts
- Keep your token secure and don't share it publicly
- The token is only used for model authentication

## 📚 Additional Resources

- [FinGPT GitHub Repository](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT%5FForecaster)
- [Hugging Face Model Page](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## ✅ Ready to Use

Once setup is complete, you can:

1. Generate forecasts for any DOW30 company
2. Use the interactive chat mode
3. Integrate the forecaster into your own applications
4. Customize prompts for specific use cases

Happy forecasting! 📈
