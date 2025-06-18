import styled from "styled-components";
import { useState, useRef, useEffect } from "react";
import { Send, Upload, X, Users, LogOut } from "react-feather";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { theme } from "../styles/theme";
import {
  sendChatMessage,
  uploadDocument,
  logout,
  callLlmTool,
} from "../utils/api";
import { useNavigate } from "react-router-dom";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: ${theme.colors.background};
  position: relative;
  font-family: ${theme.fonts.body};
`;

const ChatSection = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: ${theme.colors.background};
  position: relative;
  width: 100%;
  max-width: 1000px;
  margin: 0 auto;
  padding: 1rem;
  box-sizing: border-box;
  min-height: 0;
  overflow: hidden;
`;

const ChatHeader = styled.div`
  padding: 1rem;
  color: ${theme.colors.primary};
  font-weight: 600;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2px solid ${theme.colors.border};
  background: ${theme.colors.surface};
  margin-bottom: 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

  h1 {
    font-size: 1.5rem;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  margin-bottom: 1rem;
  border-radius: 8px;
  scrollbar-width: thin;
  scrollbar-color: ${theme.colors.primary} ${theme.colors.surface};
  min-height: 0;
  max-height: calc(100vh - 300px);
  height: calc(100vh - 300px);
  display: flex;
  flex-direction: column;

  &::-webkit-scrollbar {
    width: 8px;
    display: block;
  }

  &::-webkit-scrollbar-track {
    background: ${theme.colors.surface};
  }

  &::-webkit-scrollbar-thumb {
    background-color: ${theme.colors.primary};
    border-radius: 4px;
    border: 2px solid ${theme.colors.surface};
  }
`;

const MessageWrapper = styled.div`
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  position: relative;
`;

const Avatar = styled.div`
  width: 2.5rem;
  height: 2.5rem;
  background-color: ${(props) =>
    props.role === "user" ? theme.colors.user : theme.colors.bot};
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.875rem;
  flex-shrink: 0;
  border-radius: 50%;
  text-transform: uppercase;
`;

const LoadingAvatar = styled(Avatar)`
  background-color: ${theme.colors.bot};
  animation: pulse 1.5s infinite;

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.6;
    }
  }
`;

const MessageContent = styled.div`
  background: ${(props) =>
    props.role === "user" ? theme.colors.background : "#f8f9fa"};
  padding: 1rem;
  border-radius: 8px;
  flex: 1;
  color: ${theme.colors.text.primary};
  font-size: 0.95rem;
  border: 1px solid ${theme.colors.border};
  position: relative;

  p {
    margin: 0 0 0.5rem 0;
    line-height: 1.6;
  }

  p:last-child {
    margin-bottom: 0;
  }

  code {
    background: ${theme.colors.surface};
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: ${theme.fonts.mono};
    font-size: 0.9em;
    border: 1px solid ${theme.colors.border};
  }

  pre {
    background: ${theme.colors.surface};
    padding: 1rem;
    border-radius: 4px;
    overflow-x: auto;
    border: 1px solid ${theme.colors.border};

    code {
      background: none;
      border: none;
      padding: 0;
    }
  }
`;

const ChatInput = styled.div`
  padding: 1rem;
  background: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: 8px;
  position: relative;
`;

const InputForm = styled.form`
  display: flex;
  gap: 0.5rem;
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
  padding: 0.75rem;
  background: ${theme.colors.background};
  color: ${theme.colors.text.primary};
  border: 1px solid ${theme.colors.border};
  border-radius: 6px;
  font-size: 0.95rem;
  font-family: ${theme.fonts.body};
  resize: vertical;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: ${theme.colors.primary};
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }

  &::placeholder {
    color: ${theme.colors.text.secondary};
  }
`;

const SendButton = styled.button`
  background: ${theme.colors.primary};
  color: white;
  border: none;
  border-radius: 6px;
  padding: 0.75rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.2s ease;
  height: 48px;

  &:hover:not(:disabled) {
    background: ${theme.colors.primaryHover};
  }

  &:disabled {
    background: ${theme.colors.text.secondary};
    cursor: not-allowed;
  }
`;

const UploadButton = styled.button`
  background: transparent;
  color: ${theme.colors.primary};
  border: 1px solid ${theme.colors.primary};
  border-radius: 6px;
  padding: 0.75rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  height: 48px;

  &:hover:not(:disabled) {
    background: ${theme.colors.primary};
    color: white;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const FileUploadSection = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0;
`;

const FileInfo = styled.div`
  font-size: 0.875rem;
  color: ${theme.colors.text.secondary};
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const RemoveFileButton = styled.button`
  background: transparent;
  border: none;
  color: ${theme.colors.text.error};
  cursor: pointer;
  display: flex;
  align-items: center;
  padding: 0.25rem;

  &:hover {
    background: rgba(220, 53, 69, 0.1);
    border-radius: 4px;
  }
`;

const ControlsSection = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem;
  background: ${theme.colors.background};
  border: 1px solid ${theme.colors.border};
  margin-bottom: 1rem;
  border-radius: 6px;
`;

const ToggleLabel = styled.label`
  font-size: 0.9rem;
  color: ${theme.colors.text.primary};
  margin-left: 0.5rem;
  flex: 1;
  font-weight: 500;
`;

const ToggleSwitch = styled.div`
  position: relative;
  width: 44px;
  height: 24px;
  background: ${(props) =>
    props.checked ? theme.colors.primary : theme.colors.text.secondary};
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s;

  &::after {
    content: "";
    position: absolute;
    top: 2px;
    left: ${(props) => (props.checked ? "22px" : "2px")};
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: all 0.3s;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
`;

const AdminHeader = styled.div`
  padding: 0.5rem;
  background: rgba(220, 53, 69, 0.1);
  color: ${theme.colors.text.error};
  border: 1px solid rgba(220, 53, 69, 0.3);
  border-radius: 6px;
  margin-bottom: 1rem;
  font-size: 0.875rem;
  text-align: center;
  font-weight: 500;
`;

const UserInfoBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  background: ${theme.colors.surface};
  border-top: 1px solid ${theme.colors.border};
  margin-top: 1rem;
  font-size: 0.875rem;
  color: ${theme.colors.text.secondary};
  border-radius: 0 0 8px 8px;
`;

const UserBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;

  span.role {
    color: ${(props) =>
      props.isAdmin ? theme.colors.text.error : theme.colors.primary};
    font-weight: 600;
    text-transform: capitalize;
  }
`;

const LogoutButton = styled.button`
  background: transparent;
  border: none;
  color: ${theme.colors.text.secondary};
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-size: 0.875rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  transition: all 0.2s ease;

  &:hover {
    color: ${theme.colors.text.primary};
    background: rgba(0, 0, 0, 0.05);
  }
`;

export default function ChatPage() {
  const navigate = useNavigate();
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome to FinGPT! 💰\n\nI'm your personal finance assistant. I can help you with:\n\n• Budget planning and expense tracking\n• Investment analysis and portfolio optimization\n• Financial document processing\n• Market insights and economic trends\n• Tax planning strategies\n• Retirement planning\n\nHow can I help you with your finances today?",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [document, setDocument] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // New state for role-based features
  const [user, setUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [isAdminModeEnabled, setIsAdminModeEnabled] = useState(false);

  // Check auth on load
  useEffect(() => {
    const storedUser = localStorage.getItem("user");

    if (!storedUser) {
      navigate("/");
      return;
    }

    try {
      const parsedUser = JSON.parse(storedUser);
      setUser(parsedUser);
      setIsAdmin(parsedUser.role === "admin");

      // Show different messages based on user role
      if (parsedUser.role === "admin") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "🔧 **Admin Access Detected**\n\nYou can toggle between standard Finance mode and Admin function mode using the toggle above.\n\nIn Admin mode, you can:\n- Create new user accounts\n- Delete existing users\n- Modify user information\n- Get user data\n\nJust describe what you want to do in plain English.",
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "💡 **Tip**: You can toggle between standard Finance mode and Advanced mode using the toggle above.\n\nIn Advanced mode, you can perform additional account management operations and access more sophisticated financial tools.\n\nJust describe what you want to do in plain English.",
          },
        ]);
      }
    } catch (error) {
      console.error("Error parsing user data:", error);
      navigate("/");
    }
  }, [navigate]);

  // Auto-scroll to the bottom of the chat when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleLogout = async () => {
    try {
      await logout();
      localStorage.removeItem("user");
      navigate("/");
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  const handleLlmToolCall = async () => {
    if (!user) return;

    setIsLoading(true);

    const newMessage = { role: "user", content: message };

    // Add user message to the conversation
    setMessages((prev) => [...prev, newMessage]);
    setMessage("");

    try {
      // Get the last few messages for context (up to 10 messages)
      const conversationHistory = messages.slice(-10);

      // Add the new user message to conversation history
      conversationHistory.push(newMessage);

      // Call the LLM tool calling endpoint with the user's role
      const response = await callLlmTool(
        newMessage.content,
        conversationHistory,
        user.role
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.llm_response || response.response,
        },
      ]);
    } catch (error) {
      console.error("Error with LLM tool call:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: Unable to process request. (${error.message})`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim() || isLoading) return;

    // Check if admin mode is enabled for this user
    if (isAdminModeEnabled) {
      await handleLlmToolCall();
      return;
    }

    setIsLoading(true);

    const newMessage = { role: "user", content: message };

    // Add user message to the conversation
    setMessages((prev) => [...prev, newMessage]);
    setMessage("");

    try {
      const response = await sendChatMessage(
        newMessage.content,
        sessionId,
        "gemini-2.0-flash"
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.answer,
        },
      ]);
    } catch (error) {
      console.error("Error sending message:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: Unable to process your request. (${error.message})`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!document) return;

    setIsUploading(true);

    try {
      const response = await uploadDocument(document);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `✅ **Document Uploaded Successfully**\n\nFile: ${document.name}\nPages processed: ${response.pages_processed}\n\nI can now answer questions about this document. What would you like to know?`,
        },
      ]);

      // Clear the file selection after successful upload
      setDocument(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      console.error("Error uploading document:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `❌ **Upload Failed**\n\nError: ${error.message}\n\nPlease try again with a different file.`,
        },
      ]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    setDocument(file);
  };

  const clearChat = () => {
    setMessages([
      {
        role: "assistant",
        content:
          "Welcome to FinGPT! 💰\n\nI'm your personal finance assistant. How can I help you with your finances today?",
      },
    ]);
  };

  const toggleAdminMode = () => {
    setIsAdminModeEnabled(!isAdminModeEnabled);
  };

  return (
    <Container>
      <ChatSection>
        <ChatHeader>
          <h1>💰 FinGPT</h1>
          <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
            <button
              onClick={clearChat}
              style={{
                background: "transparent",
                border: `1px solid ${theme.colors.border}`,
                color: theme.colors.text.secondary,
                padding: "0.5rem 1rem",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "0.875rem",
              }}
            >
              Clear Chat
            </button>
          </div>
        </ChatHeader>

        {isAdmin && isAdminModeEnabled && (
          <AdminHeader>
            🔧 Admin Mode Active - You can now manage users and perform
            administrative tasks
          </AdminHeader>
        )}

        <ControlsSection>
          <div style={{ display: "flex", alignItems: "center" }}>
            <ToggleSwitch
              checked={isAdminModeEnabled}
              onClick={toggleAdminMode}
            />
            <ToggleLabel>
              {isAdminModeEnabled
                ? isAdmin
                  ? "Admin Functions"
                  : "Advanced Mode"
                : "Finance Mode"}
            </ToggleLabel>
          </div>
        </ControlsSection>

        <ChatMessages>
          {messages.map((msg, index) => (
            <MessageWrapper key={index}>
              <Avatar role={msg.role}>
                {msg.role === "user" ? user?.username?.charAt(0) || "U" : "F"}
              </Avatar>
              <MessageContent role={msg.role}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>
              </MessageContent>
            </MessageWrapper>
          ))}

          {isLoading && (
            <MessageWrapper>
              <LoadingAvatar>F</LoadingAvatar>
              <MessageContent role="assistant">
                <p>Analyzing your request...</p>
              </MessageContent>
            </MessageWrapper>
          )}

          <div ref={messagesEndRef} />
        </ChatMessages>

        <ChatInput>
          {document && (
            <FileUploadSection>
              <FileInfo>
                📄 {document.name} ({(document.size / 1024).toFixed(1)} KB)
                <RemoveFileButton onClick={() => setDocument(null)}>
                  <X size={16} />
                </RemoveFileButton>
              </FileInfo>
              <UploadButton
                onClick={handleFileUpload}
                disabled={isUploading}
                style={{ fontSize: "0.875rem", padding: "0.5rem 1rem" }}
              >
                {isUploading ? "Uploading..." : "Upload"}
              </UploadButton>
            </FileUploadSection>
          )}

          <InputForm onSubmit={handleSendMessage}>
            <TextareaWrapper>
              <StyledTextarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask me about your finances, budgeting, investments, or upload a document..."
                disabled={isLoading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage(e);
                  }
                }}
              />
            </TextareaWrapper>

            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileSelect}
              accept=".pdf,.doc,.docx,.txt"
              style={{ display: "none" }}
            />

            <UploadButton
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || isUploading}
            >
              <Upload size={18} />
            </UploadButton>

            <SendButton type="submit" disabled={!message.trim() || isLoading}>
              <Send size={18} />
            </SendButton>
          </InputForm>
        </ChatInput>

        <UserInfoBar>
          <UserBadge isAdmin={isAdmin}>
            <Users size={16} />
            <span>{user?.username}</span>
            <span className="role">({user?.role})</span>
          </UserBadge>
          <LogoutButton onClick={handleLogout}>
            <LogOut size={16} />
            Logout
          </LogoutButton>
        </UserInfoBar>
      </ChatSection>
    </Container>
  );
}
