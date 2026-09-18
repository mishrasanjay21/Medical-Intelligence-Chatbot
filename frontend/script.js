const API_URL = "http://127.0.0.1:8000/ask";
const questionForm = document.getElementById("questionForm");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const chatBox = document.getElementById("chatBox");

function addMessage(message, type) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}`;
    if (type === "bot") {
        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.textContent = "MI";
        avatar.setAttribute("aria-hidden", "true");
        messageDiv.appendChild(avatar);
    }
    const stack = document.createElement("div");
    stack.className = "message-stack";
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = type === "bot" ? "Assistant" : "You";
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message;
    stack.append(label, content);
    messageDiv.appendChild(stack);
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function setLoading(isLoading) {
    sendButton.disabled = isLoading;
    sendButton.querySelector("span").textContent = isLoading ? "Thinking..." : "Ask";
}

async function askQuestion(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || sendButton.disabled) return;
    addMessage(question, "user");
    questionInput.value = "";
    setLoading(true);
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, top_k: 10 })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "The backend could not answer.");
        addMessage(`${data.answer}\n\n${data.disclaimer}`, "bot");
    } catch (error) {
        console.error(error);
        addMessage("I could not connect to the backend. Please make sure the FastAPI server is running on port 8000.", "bot");
    } finally {
        setLoading(false);
        questionInput.focus();
    }
}

questionForm.addEventListener("submit", askQuestion);
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        questionForm.requestSubmit();
    }
});
