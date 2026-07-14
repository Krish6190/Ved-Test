import React from 'react';
import './ModeButton.css';

const ModeButton = ({ label, icon, isActive, onClick }) => {
  return (
    <button 
      className={`mode-btn ${isActive ? 'active' : ''} ${label.toLowerCase()}-mode`} 
      onClick={onClick}
    >
      <span className="mode-icon">{icon}</span>
      <span className="mode-label">{label}</span>
    </button>
  );
};

export default ModeButton;
