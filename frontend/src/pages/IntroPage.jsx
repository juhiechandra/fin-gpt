import { useState } from "react";
import { useNavigate } from "react-router-dom";
import styled from "styled-components";
import { theme } from "../styles/theme";
import { login } from "../utils/api";

const Container = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background-color: ${theme.colors.background};
  color: ${theme.colors.text.primary};
  font-family: ${theme.fonts.body};
  padding: 2rem;
`;

const LoginCard = styled.div`
  width: 100%;
  max-width: 400px;
  background-color: ${theme.colors.surface};
  border: 1px solid ${theme.colors.border};
  border-radius: 8px;
  padding: 2rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  color: ${theme.colors.primary};
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  font-weight: 600;
`;

const Subtitle = styled.p`
  color: ${theme.colors.text.secondary};
  text-align: center;
  margin-bottom: 2rem;
  font-size: 1.1rem;
`;

const Description = styled.div`
  margin-bottom: 2rem;
  padding: 1rem;
  background-color: ${theme.colors.background};
  border-left: 3px solid ${theme.colors.primary};
  font-size: 0.95rem;
  color: ${theme.colors.text.secondary};
  line-height: 1.6;
  border-radius: 0 4px 4px 0;

  p {
    margin-bottom: 0.5rem;
  }

  p:last-child {
    margin-bottom: 0;
  }
`;

const LoginForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  color: ${theme.colors.text.primary};
  font-size: 0.9rem;
  font-weight: 500;
`;

const Input = styled.input`
  background-color: ${theme.colors.background};
  color: ${theme.colors.text.primary};
  border: 1px solid ${theme.colors.border};
  border-radius: 4px;
  padding: 0.75rem;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: ${theme.colors.primary};
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }
`;

const LoginButton = styled.button`
  background-color: ${theme.colors.primary};
  color: white;
  border: none;
  border-radius: 4px;
  padding: 0.75rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s ease;
  margin-top: 0.5rem;

  &:hover {
    background-color: ${theme.colors.primaryHover};
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const ErrorMessage = styled.div`
  color: ${theme.colors.text.error};
  font-size: 0.875rem;
  padding: 0.5rem;
  border-radius: 4px;
  background-color: rgba(220, 53, 69, 0.1);
  border: 1px solid rgba(220, 53, 69, 0.3);
`;

export default function IntroPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!username || !password) {
      setError("Username and password are required");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const response = await login(username, password);

      // Store user info in localStorage
      localStorage.setItem(
        "user",
        JSON.stringify({
          id: response.user_id,
          username: response.username,
          role: response.role,
        })
      );

      // Navigate to chat page
      navigate("/chat");
    } catch (error) {
      setError(error.message || "Invalid credentials");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container>
      <LoginCard>
        <Title>FinGPT</Title>
        <Subtitle>Your Personal Finance Assistant</Subtitle>

        <Description>
          <p>
            <strong>Welcome to FinGPT!</strong>
          </p>
          <p>
            Get personalized financial advice, analyze your spending, create
            budgets, and make informed investment decisions with the help of AI.
          </p>
          <p>Features include:</p>
          <ul>
            <li>Personal budget planning</li>
            <li>Investment analysis</li>
            <li>Financial document processing</li>
            <li>Market insights and trends</li>
          </ul>
        </Description>

        <LoginForm onSubmit={handleLogin}>
          <InputGroup>
            <Label>Username</Label>
            <Input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={isLoading}
              placeholder="Enter your username"
            />
          </InputGroup>
          <InputGroup>
            <Label>Password</Label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isLoading}
              placeholder="Enter your password"
            />
          </InputGroup>
          {error && <ErrorMessage>{error}</ErrorMessage>}
          <LoginButton type="submit" disabled={isLoading}>
            {isLoading ? "Signing in..." : "Sign In"}
          </LoginButton>
        </LoginForm>
      </LoginCard>
    </Container>
  );
}
