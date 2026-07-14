import React, { useState } from 'react';
import './ChatInput.css';

const ChatInput = ({ onSendMessage }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    onSendMessage(inputValue);
    setInputValue('');
  };

  return (
    <form className="chat-input-container" onSubmit={handleSubmit}>
      <button type="button" className="action-icon-btn file-btn" title="Upload File">
        ➕
      </button>
      
      <input 
        type="text" 
        className="prompt-input" 
        placeholder="Ask Ved anything..." 
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
      />
      
      <button type="button" className="action-icon-btn mic-btn" title="Record Voice">
        🎙️
      </button>
    </form>
  );
};

export default ChatInput;
