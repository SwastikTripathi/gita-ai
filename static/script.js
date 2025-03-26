let conversationHistory = [];
let promptCount = parseInt(localStorage.getItem('promptCount')) || 0;
let verseShuffleInterval = null;
let currentTypeInterval = null;
let currentFetchController = null;
let currentRotation = 0;
let interimInterval = null;
let progressInterval = null;
let currentVerse = null;
let responseReceived = false;
let verseTimeout;

const interimMessages = [
    "Searching the scriptures...",
    "Consulting the sages...",
    "Finding wisdom for you...",
    "Reflecting on your query...",
    "Skimming the old books...",
    "Texting the wise folks...",
    "Digging up some answers...",
    "Mulling over your question...",
    "Checking my notes real quick...",
    "Asking around the brain trust...",
    "Rifling through the files...",
    "Puzzling this one out..."
];

const initialInput = document.getElementById('initial-input');
const initialSendButton = document.getElementById('initial-send-button');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const sendButtonImg = sendButton.querySelector('img');
const conversationDiv = document.getElementById('conversation');
const clearBtn = document.getElementById('clear-btn');
const clearBtnImg = clearBtn.querySelector('img');
const modal = document.getElementById('modal');
const modalClearBtn = document.getElementById('modal-clear-btn');
const themeToggle = document.getElementById('theme-toggle-btn');
const helpBtn = document.getElementById('help-btn');
const helpBtnImg = helpBtn.querySelector('img');
const helpModal = document.getElementById('help-modal');
const helpCloseBtn = document.getElementById('help-close-btn');
const inputArea = document.getElementById('input-area');
const initialScreen = document.querySelector('.initial-screen');
const initialGreeting = document.getElementById('initial-greeting');
const welcomeLine2 = document.getElementById('welcome-line2');
const waterdropBtn = document.getElementById('waterdrop-btn');
const slidingMenu = document.getElementById('sliding-menu');
const verseBtn = document.querySelector('.verse-btn');
const verseDisplay = document.getElementById('verse-display');

waterdropBtn.addEventListener('click', () => {
    const isOpen = slidingMenu.classList.contains('open');
    const menuWidth = slidingMenu.offsetWidth;
    if (isOpen) {
        slidingMenu.classList.remove('open');
        waterdropBtn.style.right = '0';
    } else {
        slidingMenu.classList.add('open');
        waterdropBtn.style.right = `${menuWidth}px`;
    }
});

document.addEventListener('click', (e) => {
    if (!slidingMenu.contains(e.target) && !waterdropBtn.contains(e.target) && slidingMenu.classList.contains('open')) {
        slidingMenu.classList.remove('open');
        waterdropBtn.style.right = '0';
    }
});

async function fetchRandomVerse() {
    const response = await fetch('/api/random_verse');
    const data = await response.json();
    currentVerse = data;
    const formattedVerse = String(data.verse || "").split(' | ').join(' |<br>');
    data.meaning = data.meaning.replace(/^\d+(\.\d+)?\s*/, '');
    document.getElementById('verse-content').innerHTML = `
        <strong>${formattedVerse}</strong><br><br>${data.meaning}
    `;
}

function showVerseDisplay() {
    verseDisplay.classList.add('open');
    if (!currentVerse) fetchRandomVerse();
    verseShuffleInterval = setInterval(fetchRandomVerse, 15000);
}

function hideVerseDisplay() {
    verseDisplay.classList.remove('open');
    if (verseShuffleInterval) {
        clearInterval(verseShuffleInterval);
        verseShuffleInterval = null;
    }
}

verseBtn.addEventListener('click', () => {
    verseDisplay.classList.contains('open') ? hideVerseDisplay() : showVerseDisplay();
});

document.getElementById('next-verse').addEventListener('click', fetchRandomVerse);
document.getElementById('prev-verse').addEventListener('click', fetchRandomVerse);

function getTimeOfDay() {
    const now = new Date();
    const hours = now.getHours();
    if (hours >= 5 && hours < 12) return "morning";
    else if (hours >= 12 && hours < 17) return "afternoon";
    else if (hours >= 17 && hours < 22) return "evening";
    else return "night";
}

function setWelcomeMessage() {
    const timeOfDay = getTimeOfDay();
    welcomeLine2.textContent = `How can I help you this ${timeOfDay}?`;
}

function setPrompt(prompt) {
    initialInput.value = prompt;
    sendMessage('initial');
}

function getTimestamp() {
    const now = new Date();
    return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
}

function scrollToBottom() {
    conversationDiv.scrollTop = conversationDiv.scrollHeight;
}

function updateNavbarBorderPosition() {
    const navbar = document.querySelector('.navbar');
    const navbarBorder = document.querySelector('.navbar-border');
    navbarBorder.style.top = `${navbar.offsetHeight}px`;
}

function stopGeneration() {
    if (currentTypeInterval) {
        clearInterval(currentTypeInterval);
        currentTypeInterval = null;
        const lastAiMessage = conversationDiv.querySelector('.message-wrapper.ai:last-child .message');
        if (lastAiMessage) {
            const textSpan = lastAiMessage.querySelector('.text');
            if (textSpan) {
                lastAiMessage.innerHTML = `<div class="text">${marked.parse(textSpan.textContent)}</div><span class="timestamp">${getTimestamp()}</span>`;
            }
        }
    }
    if (currentFetchController) {
        currentFetchController.abort();
        currentFetchController = null;
    }
    if (interimInterval) {
        clearInterval(interimInterval);
        interimInterval = null;
    }
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    const typingIndicatorWrapper = conversationDiv.querySelector('.message-wrapper.ai:last-child');
    if (typingIndicatorWrapper && typingIndicatorWrapper.querySelector('#loading-spinner')) {
        conversationDiv.removeChild(typingIndicatorWrapper);
    }
    sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/arrow-dark.png?raw=true';
    sendButtonImg.alt = 'Send';
    scrollToBottom();
    userInput.focus();
    clearTimeout(verseTimeout);
    hideVerseDisplay();
}

function startInterimMessages() {
    const interimDiv = document.getElementById('interim-message');
    interimInterval = setInterval(() => {
        if (!responseReceived) {
            const randomIndex = Math.floor(Math.random() * interimMessages.length);
            interimDiv.innerHTML = `<span>${interimMessages[randomIndex]}</span>`;
        } else {
            clearInterval(interimInterval);
            interimInterval = null;
        }
    }, 10000);
}

async function sendMessage(messageSource = 'user') {
    if (promptCount >= 5) {
        const limitMessage = "You have reached the limit of 5 prompts. This is a prototype version. If you liked it and want to support it, you can through [this link](https://buymeacoffee.com/not.a.toaster).";
        const aiMessageWrapper = document.createElement('div');
        aiMessageWrapper.className = 'message-wrapper ai';
        aiMessageWrapper.innerHTML = `
            <div class="message ai">
                <div class="text">${marked.parse(limitMessage)}</div>
                <span class="timestamp">${getTimestamp()}</span>
            </div>
        `;
        conversationDiv.appendChild(aiMessageWrapper);
        scrollToBottom();
        return;
    }

    stopGeneration();

    const message = messageSource === 'initial' ? initialInput.value.trim() : userInput.value.trim();
    if (!message) return;

    if (currentRotation === 0) currentRotation = -90;
    else currentRotation -= 360;
    sendButton.style.transform = `rotate(${currentRotation}deg)`;

    const userId = Date.now().toString();
    const userMessageWrapper = document.createElement('div');
    userMessageWrapper.className = 'message-wrapper user';
    userMessageWrapper.dataset.id = userId;
    userMessageWrapper.innerHTML = `
        <div class="message user">
            <div class="text">${message}</div>
            <span class="timestamp">${getTimestamp()}</span>
        </div>
        <button class="edit-btn"><img src="${document.documentElement.getAttribute('data-theme') === 'dark' ? 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/edit-light.png?raw=true' : 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/edit-dark.png?raw=true'}" alt="Edit"></button>
    `;
    conversationDiv.appendChild(userMessageWrapper);
    conversationHistory.push({ id: userId, role: 'user', content: message });

    initialScreen.style.display = 'none';
    conversationDiv.style.display = 'block';
    inputArea.style.display = 'flex';
    if (messageSource === 'initial') initialInput.value = '';
    else userInput.value = '';
    userInput.style.height = 'auto';
    scrollToBottom();

    const typingIndicatorWrapper = document.createElement('div');
    typingIndicatorWrapper.className = 'message-wrapper ai';
    typingIndicatorWrapper.innerHTML = `
        <div class="message ai">
            <div id="loading-spinner">
                <img src="/static/img/flute.gif" alt="Loading flute animation">
                <p id="interim-message"><span>Processing...</span></p>
                <progress id="wait-progress" max="100" value="0"></progress>
            </div>
        </div>
    `;
    conversationDiv.appendChild(typingIndicatorWrapper);
    scrollToBottom();
    sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/stop-dark.png?raw=true';
    sendButtonImg.alt = 'Stop';

    responseReceived = false;
    startInterimMessages();

    const progressBar = document.getElementById('wait-progress');
    let progress = 0;
    const maxWaitTime = 120000;
    progressInterval = setInterval(() => {
        if (!responseReceived) {
            progress += (1000 / maxWaitTime) * 100;
            progressBar.value = Math.min(progress, 100);
        } else {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }, 1000);

    verseTimeout = setTimeout(() => {
        if (!responseReceived) showVerseDisplay();
    }, 5000);

    currentFetchController = new AbortController();
    let data;
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message, id: userId }),
            signal: currentFetchController.signal
        });
        data = await response.json();
        responseReceived = true;
        clearTimeout(verseTimeout);
        hideVerseDisplay();
        clearInterval(interimInterval);
        clearInterval(progressInterval);
        conversationDiv.removeChild(typingIndicatorWrapper);
    } catch (error) {
        responseReceived = true;
        clearTimeout(verseTimeout);
        hideVerseDisplay();
        clearInterval(interimInterval);
        clearInterval(progressInterval);
        conversationDiv.removeChild(typingIndicatorWrapper);
        sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/arrow-dark.png?raw=true';
        sendButtonImg.alt = 'Send';
        userInput.focus();
        return;
    }

    const aiMessageWrapper = document.createElement('div');
    aiMessageWrapper.className = 'message-wrapper ai';
    aiMessageWrapper.dataset.id = data.id;
    const aiMessage = document.createElement('div');
    aiMessage.className = 'message ai';
    const textSpan = document.createElement('span');
    textSpan.className = 'text';
    aiMessage.appendChild(textSpan);
    aiMessageWrapper.appendChild(aiMessage);
    conversationDiv.appendChild(aiMessageWrapper);
    conversationHistory.push({ id: data.id, role: 'ai', content: data.response });

    const fullResponse = data.response || "Sorry, I couldn’t find a suitable verse from the Bhagavad Gita for your query.";
    let i = 0;
    const isAtBottom = Math.abs(conversationDiv.scrollHeight - conversationDiv.scrollTop - conversationDiv.clientHeight) < 1;
    let autoScroll = isAtBottom;
    const scrollListener = () => {
        autoScroll = false;
        conversationDiv.removeEventListener('scroll', scrollListener);
    };
    conversationDiv.addEventListener('scroll', scrollListener);

    currentTypeInterval = setInterval(() => {
        if (i < fullResponse.length) {
            textSpan.innerHTML = fullResponse.slice(0, i + 1) + '<span class="cursor"></span>';
            i++;
            if (autoScroll) scrollToBottom();
        } else {
            clearInterval(currentTypeInterval);
            currentTypeInterval = null;
            aiMessage.innerHTML = `<div class="text">${marked.parse(fullResponse)}</div><span class="timestamp">${getTimestamp()}</span>`;
            sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/arrow-dark.png?raw=true';
            sendButtonImg.alt = 'Send';
            if (autoScroll) scrollToBottom();
            userInput.focus();
            document.getElementById('notification-sound').play();
            promptCount++;
            localStorage.setItem('promptCount', promptCount);
        }
    }, 3);
}

sendButton.addEventListener('click', () => sendMessage('user'));
initialSendButton.addEventListener('click', () => sendMessage('initial'));

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage('user');
    }
});

initialInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage('initial');
    }
});

userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = `${userInput.scrollHeight}px`;
    userInput.style.overflowY = userInput.scrollHeight > 150 ? 'auto' : 'hidden';
});

initialInput.addEventListener('input', () => {
    initialInput.style.height = '60px';
    initialInput.style.overflowY = initialInput.scrollHeight > 60 ? 'auto' : 'hidden';
});

conversationDiv.addEventListener('click', (e) => {
    if (e.target.closest('.edit-btn')) {
        const wrapper = e.target.closest('.message-wrapper');
        const messageDiv = wrapper.querySelector('.message');
        const textDiv = messageDiv.querySelector('.text');
        const timestampSpan = messageDiv.querySelector('.timestamp');
        const currentText = textDiv.textContent;
        const input = document.createElement('textarea');
        input.value = currentText;
        input.className = 'edit-input';
        textDiv.innerHTML = '';
        textDiv.appendChild(input);
        input.focus();

        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = `${input.scrollHeight}px`;
            input.style.overflowY = input.scrollHeight > 150 ? 'auto' : 'hidden';
        });

        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const newText = input.value.trim();
                if (newText) {
                    textDiv.textContent = newText;
                    messageDiv.appendChild(timestampSpan);
                    stopGeneration();

                    const index = conversationHistory.findIndex(item => item.id === wrapper.dataset.id);
                    if (index !== -1) {
                        conversationHistory[index].content = newText;
                    }

                    let nextWrapper = wrapper.nextElementSibling;
                    while (nextWrapper) {
                        const toRemove = nextWrapper;
                        nextWrapper = nextWrapper.nextElementSibling;
                        conversationDiv.removeChild(toRemove);
                        const removeIndex = conversationHistory.findIndex(item => item.id === toRemove.dataset.id);
                        if (removeIndex !== -1) conversationHistory.splice(removeIndex, 1);
                    }

                    await fetch('/api/update_message', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id: wrapper.dataset.id, content: newText })
                    });

                    const typingIndicatorWrapper = document.createElement('div');
                    typingIndicatorWrapper.className = 'message-wrapper ai';
                    typingIndicatorWrapper.innerHTML = `
                        <div class="message ai">
                            <div id="loading-spinner">
                                <img src="/static/img/flute.gif" alt="Loading flute animation">
                                <p id="interim-message"><span>Processing...</span></p>
                                <progress id="wait-progress" max="100" value="0"></progress>
                            </div>
                        </div>
                    `;
                    conversationDiv.appendChild(typingIndicatorWrapper);
                    scrollToBottom();
                    sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/stop-dark.png?raw=true';
                    sendButtonImg.alt = 'Stop';

                    responseReceived = false;
                    startInterimMessages();

                    const progressBar = document.getElementById('wait-progress');
                    let progress = 0;
                    const maxWaitTime = 120000;
                    progressInterval = setInterval(() => {
                        if (!responseReceived) {
                            progress += (1000 / maxWaitTime) * 100;
                            progressBar.value = Math.min(progress, 100);
                        } else {
                            clearInterval(progressInterval);
                            progressInterval = null;
                        }
                    }, 1000);

                    verseTimeout = setTimeout(() => {
                        if (!responseReceived) showVerseDisplay();
                    }, 5000);

                    currentFetchController = new AbortController();
                    const response = await fetch('/api/regenerate_after', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id: wrapper.dataset.id }),
                        signal: currentFetchController.signal
                    });
                    const data = await response.json();
                    responseReceived = true;
                    clearTimeout(verseTimeout);
                    hideVerseDisplay();
                    clearInterval(interimInterval);
                    clearInterval(progressInterval);
                    conversationDiv.removeChild(typingIndicatorWrapper);

                    const aiMessageWrapper = document.createElement('div');
                    aiMessageWrapper.className = 'message-wrapper ai';
                    aiMessageWrapper.dataset.id = data.id;
                    const aiMessage = document.createElement('div');
                    aiMessage.className = 'message ai';
                    const textSpan = document.createElement('span');
                    textSpan.className = 'text';
                    aiMessage.appendChild(textSpan);
                    aiMessageWrapper.appendChild(aiMessage);
                    conversationDiv.appendChild(aiMessageWrapper);
                    conversationHistory.push({ id: data.id, role: 'ai', content: data.response });

                    const fullResponse = data.response || "Sorry, I couldn’t find a suitable verse.";
                    let i = 0;
                    const isAtBottom = Math.abs(conversationDiv.scrollHeight - conversationDiv.scrollTop - conversationDiv.clientHeight) < 1;
                    let autoScroll = isAtBottom;
                    const scrollListener = () => {
                        autoScroll = false;
                        conversationDiv.removeEventListener('scroll', scrollListener);
                    };
                    conversationDiv.addEventListener('scroll', scrollListener);

                    currentTypeInterval = setInterval(() => {
                        if (i < fullResponse.length) {
                            textSpan.innerHTML = fullResponse.slice(0, i + 1) + '<span class="cursor"></span>';
                            i++;
                            if (autoScroll) scrollToBottom();
                        } else {
                            clearInterval(currentTypeInterval);
                            currentTypeInterval = null;
                            aiMessage.innerHTML = `<div class="text">${marked.parse(fullResponse)}</div><span class="timestamp">${getTimestamp()}</span>`;
                            sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/arrow-dark.png?raw=true';
                            sendButtonImg.alt = 'Send';
                            if (autoScroll) scrollToBottom();
                            userInput.focus();
                            document.getElementById('notification-sound').play();
                            promptCount++;
                            localStorage.setItem('promptCount', promptCount);
                        }
                    }, 3);
                } else {
                    textDiv.textContent = currentText;
                    messageDiv.appendChild(timestampSpan);
                }
            } else if (e.key === 'Escape') {
                textDiv.textContent = currentText;
                messageDiv.appendChild(timestampSpan);
            }
        });

        input.addEventListener('blur', () => {
            if (textDiv.contains(input)) {
                textDiv.textContent = currentText;
                messageDiv.appendChild(timestampSpan);
            }
        });
    }
});

document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        modal.style.display = 'flex';
        modal.focus();
    }
});

clearBtn.addEventListener('click', () => {
    modal.style.display = 'flex';
    modal.focus();
});

function clearConversation() {
    stopGeneration();
    fetch('/api/clear', { method: 'POST' });
    conversationDiv.innerHTML = `
        <div class="message-wrapper ai">
            <div class="message">
                <div class="text"><em>Disclaimer: I am Gita AI</em></div>
                <span class="timestamp"></span>
            </div>
        </div>
    `;
    conversationHistory = [];
    initialScreen.style.display = 'flex';
    conversationDiv.style.display = 'none';
    inputArea.style.display = 'none';
    let modAngle = ((currentRotation % 360) + 360) % 360;
    if (modAngle !== 0) {
        currentRotation -= modAngle;
        sendButton.style.transform = `rotate(${currentRotation}deg)`;
    }
    sendButtonImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/arrow-dark.png?raw=true';
    sendButtonImg.alt = 'Send';
    scrollToBottom();
    modal.style.display = 'none';
    setWelcomeMessage();
    initialInput.focus();
}

modalClearBtn.addEventListener('click', clearConversation);
modal.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') clearConversation();
});
modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.style.display = 'none';
});

themeToggle.addEventListener('click', () => {
    themeToggle.classList.toggle('theme-toggle--toggled');
    const isDark = themeToggle.classList.contains('theme-toggle--toggled');
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    clearBtnImg.src = isDark ? 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/new-light.png?raw=true' : 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/new-dark.png?raw=true';
    helpBtnImg.src = isDark ? 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/help-light.png?raw=true' : 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/help-dark.png?raw=true';
    document.querySelectorAll('.edit-btn img').forEach(img => img.src = isDark ? 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/edit-light.png?raw=true' : 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/edit-dark.png?raw=true');
    document.querySelector('#prev-verse .verse-arrow').src = isDark ? 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/left-arrow-light.png?raw=true' : 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/left-arrow-dark.png?raw=true';
    document.querySelector('#next-verse .verse-arrow').src = isDark ? 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/right-arrow-light.png?raw=true' : 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/right-arrow-dark.png?raw=true';
    updateNavbarBorderPosition();
});

const currentTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', currentTheme);
if (currentTheme === 'dark') {
    themeToggle.classList.add('theme-toggle--toggled');
    clearBtnImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/new-light.png?raw=true';
    helpBtnImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/help-light.png?raw=true';
    document.querySelector('#prev-verse .verse-arrow').src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/left-arrow-light.png?raw=true';
    document.querySelector('#next-verse .verse-arrow').src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/right-arrow-light.png?raw=true';
} else {
    clearBtnImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/new-dark.png?raw=true';
    helpBtnImg.src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/help-dark.png?raw=true';
    document.querySelector('#prev-verse .verse-arrow').src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/left-arrow-dark.png?raw=true';
    document.querySelector('#next-verse .verse-arrow').src = 'https://github.com/SwastikTripathi/gita-ai/blob/main/static/img/right-arrow-dark.png?raw=true';
}

helpBtn.addEventListener('click', () => helpModal.style.display = 'flex');
helpCloseBtn.addEventListener('click', () => helpModal.style.display = 'none');
helpModal.addEventListener('click', (e) => {
    if (e.target === helpModal) helpModal.style.display = 'none';
});

window.addEventListener('load', () => {
    setTimeout(() => initialInput.focus(), 100);
    updateNavbarBorderPosition();
    scrollToBottom();
    setWelcomeMessage();
});

updateNavbarBorderPosition();
window.addEventListener('resize', updateNavbarBorderPosition);
scrollToBottom();
setWelcomeMessage();
