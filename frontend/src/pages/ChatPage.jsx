import styled from "styled-components";
import { useState, useRef, useEffect } from "react";
import { Send, BarChart2, TrendingUp } from "react-feather";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Plot from "react-plotly.js";
import PropTypes from "prop-types";
import { theme } from "../styles/theme";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: ${theme.colors.background};
  font-family: ${theme.fonts.body};
  color: ${theme.colors.text.primary};
`;

const ChatSection = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: ${theme.colors.background};
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: ${theme.spacing.lg};
  box-sizing: border-box;
  min-height: 0;
  overflow: hidden;
`;

const ChatHeader = styled.div`
  padding: ${theme.spacing.lg};
  color: ${theme.colors.text.primary};
  font-weight: 600;
  display: flex;
  justify-content: center;
  align-items: center;
  border-bottom: 1px solid ${theme.colors.border};
  background: ${theme.colors.surface};
  margin-bottom: ${theme.spacing.lg};
  border-radius: 12px;
  box-shadow: ${theme.shadows.md};

  h1 {
    font-size: 1.75rem;
    margin: 0;
    display: flex;
    align-items: center;
    gap: ${theme.spacing.md};
    font-family: ${theme.fonts.heading};
    font-weight: 700;
    background: linear-gradient(
      135deg,
      ${theme.colors.primary},
      ${theme.colors.accent}
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${theme.spacing.lg};
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  margin-bottom: ${theme.spacing.lg};
  border-radius: 12px;
  box-shadow: ${theme.shadows.md};
  scrollbar-width: thin;
  scrollbar-color: ${theme.colors.primary} ${theme.colors.surface};
  min-height: 0;
  max-height: calc(100vh - 320px);
  height: calc(100vh - 320px);
  display: flex;
  flex-direction: column;

  &::-webkit-scrollbar {
    width: 6px;
    display: block;
  }

  &::-webkit-scrollbar-track {
    background: ${theme.colors.surface};
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb {
    background: linear-gradient(
      135deg,
      ${theme.colors.primary},
      ${theme.colors.accent}
    );
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: ${theme.colors.primaryHover};
  }
`;

const MessageWrapper = styled.div`
  display: flex;
  gap: ${theme.spacing.md};
  margin-bottom: ${theme.spacing.xl};
  position: relative;
  animation: slideIn 0.3s ease-out;

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

const Avatar = styled.div`
  width: 2.75rem;
  height: 2.75rem;
  background: ${(props) =>
    props.role === "user"
      ? `linear-gradient(135deg, ${theme.colors.primary}, ${theme.colors.primaryHover})`
      : `linear-gradient(135deg, ${theme.colors.bot}, ${theme.colors.success})`};
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.875rem;
  flex-shrink: 0;
  border-radius: 50%;
  box-shadow: ${theme.shadows.sm};
  border: 2px solid ${theme.colors.background};
`;

const LoadingAvatar = styled(Avatar)`
  background: linear-gradient(
    135deg,
    ${theme.colors.bot},
    ${theme.colors.success}
  );
  animation: pulse 1.5s infinite;

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.8;
      transform: scale(1.05);
    }
  }
`;

const MessageContent = styled.div`
  background: ${(props) =>
    props.isError
      ? theme.colors.error + "20"
      : props.role === "user"
      ? theme.colors.surfaceLight
      : theme.colors.surfaceHover};
  padding: ${theme.spacing.lg};
  border-radius: 16px;
  flex: 1;
  color: ${(props) =>
    props.isError ? theme.colors.text.error : theme.colors.text.primary};
  font-size: 0.95rem;
  line-height: 1.6;
  border: 1px solid
    ${(props) => (props.isError ? theme.colors.error : theme.colors.border)};
  box-shadow: ${theme.shadows.sm};
  position: relative;

  &:before {
    content: "";
    position: absolute;
    top: 1rem;
    left: -8px;
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 8px 8px 8px 0;
    border-color: transparent
      ${(props) =>
        props.role === "user"
          ? theme.colors.surfaceLight
          : theme.colors.surfaceHover}
      transparent transparent;
  }

  p {
    margin: 0 0 0.75rem 0;
    line-height: 1.6;
  }

  p:last-child {
    margin-bottom: 0;
  }

  code {
    background: ${theme.colors.surface};
    color: ${theme.colors.accent};
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-family: ${theme.fonts.mono};
    font-size: 0.875em;
    border: 1px solid ${theme.colors.border};
  }

  pre {
    background: ${theme.colors.surface};
    padding: ${theme.spacing.lg};
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid ${theme.colors.border};
    box-shadow: ${theme.shadows.sm};

    code {
      background: none;
      border: none;
      padding: 0;
      color: ${theme.colors.text.primary};
    }
  }

  h1,
  h2,
  h3,
  h4,
  h5,
  h6 {
    color: ${theme.colors.text.primary};
    margin: 1rem 0 0.5rem 0;
    font-weight: 600;
  }

  h1:first-child,
  h2:first-child,
  h3:first-child {
    margin-top: 0;
  }

  ul,
  ol {
    margin: 0.5rem 0;
    padding-left: 1.5rem;
  }

  li {
    margin: 0.25rem 0;
  }

  strong {
    color: ${theme.colors.text.primary};
    font-weight: 600;
  }

  a {
    color: ${theme.colors.primary};
    text-decoration: none;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const ChatInput = styled.div`
  padding: ${theme.spacing.lg};
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: 12px;
  box-shadow: ${theme.shadows.md};
  position: relative;
`;

const InputForm = styled.form`
  display: flex;
  gap: ${theme.spacing.md};
  align-items: end;
`;

const TextareaWrapper = styled.div`
  flex: 1;
  position: relative;
`;

const StyledTextarea = styled.textarea`
  width: 100%;
  min-height: 60px;
  max-height: 150px;
  padding: ${theme.spacing.md};
  border: 1px solid ${theme.colors.border};
  border-radius: 8px;
  font-size: 0.95rem;
  resize: none;
  outline: none;
  font-family: ${theme.fonts.body};
  background: ${theme.colors.background};
  color: ${theme.colors.text.primary};
  box-sizing: border-box;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: ${theme.colors.primary};
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  &::placeholder {
    color: ${theme.colors.text.tertiary};
  }
`;

const SendButton = styled.button`
  background: linear-gradient(
    135deg,
    ${theme.colors.primary},
    ${theme.colors.accent}
  );
  color: white;
  border: none;
  border-radius: 8px;
  padding: ${theme.spacing.md};
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  box-shadow: ${theme.shadows.sm};
  width: 48px;
  height: 48px;

  &:hover:not(:disabled) {
    background: linear-gradient(
      135deg,
      ${theme.colors.primaryHover},
      ${theme.colors.accent}
    );
    transform: translateY(-1px);
    box-shadow: ${theme.shadows.md};
  }

  &:disabled {
    background: ${theme.colors.text.tertiary};
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const ExamplesSection = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${theme.spacing.sm};
  margin-bottom: ${theme.spacing.lg};
`;

const ExampleButton = styled.button`
  background: ${theme.colors.surfaceLight};
  color: ${theme.colors.text.secondary};
  border: 1px solid ${theme.colors.border};
  border-radius: 8px;
  padding: ${theme.spacing.sm} ${theme.spacing.md};
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};

  &:hover {
    background: ${theme.colors.surfaceHover};
    color: ${theme.colors.text.primary};
    border-color: ${theme.colors.primary};
    transform: translateY(-1px);
  }
`;

const StockChart = styled.div`
  margin: ${theme.spacing.lg} 0;
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: 12px;
  padding: ${theme.spacing.lg};
  box-shadow: ${theme.shadows.md};
`;

const ConnectionStatus = styled.div`
  position: absolute;
  right: ${theme.spacing.lg};
  color: ${theme.colors.text.secondary};
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${theme.colors.success};
  animation: pulse 2s infinite;

  @keyframes pulse {
    0% {
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    70% {
      box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
    }
    100% {
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
    }
  }
`;

const StatusBar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: ${theme.spacing.sm} ${theme.spacing.lg};
  background: ${theme.colors.surface}40;
  border-top: 1px solid ${theme.colors.border};
  margin-top: ${theme.spacing.md};
  border-radius: 0 0 12px 12px;
  font-size: 0.75rem;
  color: ${theme.colors.text.tertiary};
`;

const StatusInfo = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
`;

const PoweredBy = styled.div`
  font-weight: 500;
  color: ${theme.colors.text.secondary};
`;

const LoadingMessage = styled.div`
  display: flex;
  align-items: center;
  gap: ${theme.spacing.sm};
  color: ${theme.colors.text.secondary};
  font-style: italic;
  animation: fadeIn 0.3s ease-in;

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

const LoadingDots = styled.span`
  &:after {
    content: "";
    animation: dots 1.5s infinite;
  }

  @keyframes dots {
    0%,
    20% {
      content: "";
    }
    40% {
      content: ".";
    }
    60% {
      content: "..";
    }
    80%,
    100% {
      content: "...";
    }
  }
`;

// Component to render stock chart
const StockChartComponent = ({ stockData }) => {
  if (!stockData || stockData.length === 0) return null;

  console.log("Rendering chart with data:", stockData);

  const symbols = stockData.map((stock) => stock.symbol);
  const prices = stockData.map((stock) => stock.price);

  // Generate vibrant gradient colors for better visual appeal
  const generateBarColors = (prices) => {
    return prices.map((_, index) => {
      const hue = (index * 137.5) % 360; // Golden angle for good color distribution
      return `hsl(${hue}, 70%, 60%)`;
    });
  };

  const plotData = [
    {
      x: symbols,
      y: prices,
      type: "bar",
      marker: {
        color:
          stockData.length === 1
            ? theme.colors.primary
            : generateBarColors(prices),
        line: {
          color: theme.colors.background,
          width: 2,
        },
        opacity: 0.8,
      },
      text: prices.map((price) => `$${price.toFixed(2)}`),
      textposition: "auto",
      textfont: {
        color: theme.colors.text.primary,
        size: 12,
        family: theme.fonts.body,
      },
      hovertemplate: "<b>%{x}</b><br>💰 Price: $%{y:.2f}<br><extra></extra>",
    },
  ];

  const layout = {
    title: {
      text:
        stockData.length === 1
          ? `${symbols[0]} Stock Price`
          : "📊 Stock Prices Comparison",
      font: {
        color: theme.colors.text.primary,
        size: 18,
        family: theme.fonts.heading,
      },
    },
    paper_bgcolor: theme.colors.chart.background,
    plot_bgcolor: theme.colors.chart.background,
    font: { color: theme.colors.chart.text },
    xaxis: {
      title: "Stock Symbols",
      gridcolor: theme.colors.chart.grid,
      tickfont: { color: theme.colors.chart.text, size: 12 },
      tickangle: -45, // Angle for better readability
    },
    yaxis: {
      title: "Price (USD)",
      gridcolor: theme.colors.chart.grid,
      tickfont: { color: theme.colors.chart.text, size: 12 },
      tickformat: "$.2f",
    },
    margin: { t: 70, r: 40, b: 80, l: 80 },
    showlegend: false,
    height: 450,
    transition: {
      duration: 500,
      easing: "cubic-in-out",
    },
    hoverlabel: {
      bgcolor: theme.colors.surface,
      bordercolor: theme.colors.primary,
      font: { color: theme.colors.text.primary },
    },
  };

  const config = {
    displayModeBar: true,
    modeBarButtonsToRemove: ["pan2d", "lasso2d", "select2d", "autoScale2d"],
    displaylogo: false,
    responsive: true,
  };

  return (
    <StockChart>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "12px",
          fontSize: "14px",
          color: theme.colors.text.secondary,
        }}
      >
        <BarChart2 size={16} />
        <span>Interactive Stock Chart</span>
      </div>
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        style={{ width: "100%", height: "450px" }}
        useResizeHandler={true}
      />
    </StockChart>
  );
};

StockChartComponent.propTypes = {
  stockData: PropTypes.arrayOf(
    PropTypes.shape({
      symbol: PropTypes.string.isRequired,
      price: PropTypes.number.isRequired,
      change: PropTypes.number,
    })
  ),
};

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage("");
    setIsLoading(true);

    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: "chat_session",
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const aiResponse = data.response;
      const stockData = data.stock_data; // Use actual stock data from API

      console.log("Received stock data from API:", stockData);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: aiResponse,
          stockData: stockData,
        },
      ]);
    } catch (error) {
      console.error("Error:", error);
      let errorMessage =
        "I apologize, but I'm having trouble connecting to the service. Please try again.";

      if (error.name === "TypeError" && error.message.includes("fetch")) {
        errorMessage =
          "🔌 Unable to connect to the server. Please check if the backend is running on port 8000.";
      } else if (error.message.includes("500")) {
        errorMessage =
          "🚨 Server error occurred. The AI service might be temporarily unavailable.";
      } else if (error.message.includes("timeout")) {
        errorMessage =
          "⏱️ Request timed out. The AI is taking longer than usual to respond.";
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: errorMessage,
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  const handleExampleClick = (example) => {
    setInputMessage(example);
    textareaRef.current?.focus();
  };

  const examples = [
    "📊 Compare AAPL, MSFT, and GOOGL stock prices",
    "💰 What is Tesla's current stock price?",
    "📈 Show me Apple vs Tesla comparison with charts",
    "🔍 Give me detailed AMZN stock analysis",
    "💡 What are the best tech stocks to buy now?",
    "📉 Compare NVDA and AMD performance",
  ];

  return (
    <Container>
      <ChatSection>
        <ChatHeader>
          <h1>
            <TrendingUp size={24} />
            Finance AI Assistant
          </h1>
          <ConnectionStatus>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "0.875rem",
              }}
            >
              <StatusDot />
              <span>Connected</span>
            </div>
          </ConnectionStatus>
        </ChatHeader>

        <ChatMessages>
          {messages.length === 0 && (
            <div
              style={{
                textAlign: "center",
                padding: theme.spacing.xl,
                color: theme.colors.text.secondary,
              }}
            >
              <h3
                style={{
                  margin: 0,
                  marginBottom: theme.spacing.md,
                  color: theme.colors.text.primary,
                }}
              >
                Welcome to Finance AI Assistant
              </h3>
              <p style={{ margin: 0, marginBottom: theme.spacing.lg }}>
                Ask me about stock prices, financial analysis, or investment
                advice. Try one of these examples:
              </p>
              <ExamplesSection>
                {examples.map((example, index) => (
                  <ExampleButton
                    key={index}
                    onClick={() => handleExampleClick(example)}
                  >
                    <BarChart2 size={14} />
                    {example}
                  </ExampleButton>
                ))}
              </ExamplesSection>
            </div>
          )}

          {messages.map((message, index) => (
            <div key={index}>
              <MessageWrapper>
                <Avatar role={message.role}>
                  {message.role === "user" ? "U" : "AI"}
                </Avatar>
                <MessageContent role={message.role} isError={message.isError}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </MessageContent>
              </MessageWrapper>
              {message.stockData && (
                <StockChartComponent stockData={message.stockData} />
              )}
            </div>
          ))}

          {isLoading && (
            <MessageWrapper>
              <LoadingAvatar role="assistant">AI</LoadingAvatar>
              <MessageContent role="assistant">
                <LoadingMessage>
                  Analyzing your request
                  <LoadingDots />
                </LoadingMessage>
              </MessageContent>
            </MessageWrapper>
          )}

          <div ref={messagesEndRef} />
        </ChatMessages>

        <ChatInput>
          <InputForm onSubmit={handleSendMessage}>
            <TextareaWrapper>
              <StyledTextarea
                ref={textareaRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about stock prices, financial analysis, or investment advice..."
                disabled={isLoading}
              />
            </TextareaWrapper>
            <SendButton
              type="submit"
              disabled={isLoading || !inputMessage.trim()}
            >
              <Send size={20} />
            </SendButton>
          </InputForm>
        </ChatInput>

        <StatusBar>
          <StatusInfo>
            <span>💬 Finance AI Assistant</span>
            <span>•</span>
            <span>🔒 Secure Connection</span>
            <span>•</span>
            <span>📊 Real-time Stock Data</span>
          </StatusInfo>
          <PoweredBy>Powered by FinGPT AI</PoweredBy>
        </StatusBar>
      </ChatSection>
    </Container>
  );
}
