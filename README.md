# FinGPT - Personal Finance Assistant

![FinGPT Logo](💰)

FinGPT is an AI-powered personal finance assistant that helps users manage their finances, create budgets, analyze investments, and process financial documents. Built with modern web technologies and powered by advanced language models.

## 🌟 Features

### Core Financial Features

- **Personal Budget Planning**: Create and track personalized budgets
- **Investment Analysis**: Analyze portfolios and investment opportunities
- **Financial Document Processing**: Upload and analyze financial PDFs, statements, and reports
- **Market Insights**: Get real-time market data and economic trends
- **Tax Planning**: Strategic tax planning assistance
- **Retirement Planning**: Long-term financial planning tools

### Technical Features

- **Document Upload & Processing**: Support for PDF, DOC, DOCX, and TXT files
- **Intelligent Chat Interface**: Natural language conversations about your finances
- **User Management**: Secure user accounts with role-based access
- **Session Management**: Persistent chat history and document context
- **Advanced Analytics**: Custom financial reports and data visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd fin-gpt
   ```

2. **Set up the backend**

   ```bash
   cd api
   pip install -r requirements.txt
   
   # Set up environment variables
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Set up the frontend**

   ```bash
   cd frontend
   npm install
   ```

### Environment Variables

Create a `.env` file in the `api` directory with:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
# Add other required environment variables
```

### Running the Application

1. **Start the backend**

   ```bash
   cd api
   python main.py
   ```

2. **Start the frontend** (in a new terminal)

   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**
   - Frontend: <http://localhost:5173>
   - Backend API: <http://localhost:8000>

## 🏗️ Architecture

### Backend (`/api`)

- **FastAPI**: Modern Python web framework
- **SQLite**: Local database for user data and chat history
- **FAISS**: Vector database for document search
- **LangChain**: AI/ML pipeline for document processing
- **Google Gemini**: Large language model for conversations

### Frontend (`/frontend`)

- **React 18**: Modern React with hooks
- **Vite**: Fast build tool and development server
- **Styled Components**: CSS-in-JS styling
- **Mantine**: UI component library
- **React Router**: Client-side routing

## 📖 Usage

1. **Sign In**: Create an account or sign in with existing credentials
2. **Upload Documents**: Upload financial documents for analysis
3. **Ask Questions**: Chat with FinGPT about your financial situation
4. **Get Insights**: Receive personalized financial advice and recommendations
5. **Track Progress**: Monitor your financial goals and progress over time

## 🔧 Advanced Features

### Admin Mode

- User management and account administration
- Advanced database operations
- System analytics and reporting

### Document Processing

- Automatic text extraction from financial documents
- Intelligent categorization and analysis
- Cross-document insights and comparisons

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**FinGPT** - Making personal finance accessible and intelligent through AI.
