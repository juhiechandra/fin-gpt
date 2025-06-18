# FinGPT - AI Finance Assistant

AI-powered finance chatbot with real-time stock data and document analysis.

## 🎥 Demo

<https://github.com/user-attachments/assets/fingpt.mp4>

## Features

- 🤖 **AI Chat**: Financial advice and market insights
- 📈 **Stock Data**: Real-time prices and company information  
- 📄 **Document Upload**: Analyze financial PDFs and statements
- 💼 **Portfolio Analysis**: Investment tracking and recommendations

## Quick Start

### Backend

```bash
cd api
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Access at `http://localhost:5173`

## Environment Setup

Create `api/.env`:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Tech Stack

- **Backend**: FastAPI, SQLite, FAISS, LangChain
- **Frontend**: React, Vite, Styled Components
- **AI**: Google Gemini, Document Processing

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Chat with AI
- `POST /upload` - Upload documents

## License

MIT
