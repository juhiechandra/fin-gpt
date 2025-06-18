import { ThemeProvider } from "styled-components";
import { MantineProvider } from "@mantine/core";
import { theme } from "./styles/theme";
import ChatPage from "./pages/ChatPage";

function App() {
  return (
    <ThemeProvider theme={theme}>
      <MantineProvider withGlobalStyles withNormalizeCSS>
        <ChatPage />
      </MantineProvider>
    </ThemeProvider>
  );
}

export default App;
