import React from 'react';
import './TitleBar.css';

const TitleBar = ({ currentPath }) => {
  return (
    <div className="title-bar">
      <div className="title-left">
        <span className="app-logo">🤖</span>
        <span className="app-name">VED</span>
        <span className="directory-path">{currentPath || 'C:\\Users\\default\\workspace'}</span>
      </div>
      
      <div className="window-controls">
        <button className="control-btn min-btn" aria-label="Minimize">─</button>
        <button className="control-btn close-btn" aria-label="Close">✕</button>
      </div>
    </div>
  );
};

export default TitleBar;
