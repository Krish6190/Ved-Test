import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import TitleBar from "../../components/TitleBar/TitleBar";
import ModeButton from "../../components/ModeButton/ModeButton";
import ChatInput from "../../components/ChatInput/ChatInput";
import ThreadSidebar from "../../components/ThreadSidebar/ThreadSidebar";
import { useChatManager } from "../../hooks/useChatManager";
import "./Dashboard.css";

const sidebarTransition = {
    duration: 0.3,
    ease: [0.16, 1, 0.3, 1]
};

const overlayTransition = {
    duration: 0.25
};

const Dashboard = () => {
    const [currentMode, setCurrentMode] = useState("Standard");
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const {
        threads,
        activeThreadId,
        activeMessages,
        addThread,
        deleteThread,
        renameThread,
        sendMessage,
        setActiveThreadId
    } = useChatManager();

    const modes = [
        { label: "Coder", icon: "💻" },
        { label: "Turbo", icon: "⚡" },
        { label: "Standard", icon: "🌐" },
        { label: "Hibernate", icon: "⏳" }
    ];

    return (
        <div className="dashboard-window">
            <TitleBar currentPath="C:\\Users\\krish\\OneDrive\\Desktop\\ved" />
            <div className="action-header">
                <button
                    className="menu-toggle-btn"
                    onClick={() => setSidebarOpen(true)}
                >
                    📁 Threads
                </button>
                <div className="modes-row">
                    {modes.map((mode) => (
                        <ModeButton
                            key={mode.label}
                            label={mode.label}
                            icon={mode.icon}
                            isActive={currentMode === mode.label}
                            onClick={() => setCurrentMode(mode.label)}
                        />
                    ))}
                </div>

            </div>
            <AnimatePresence>
                {sidebarOpen && (
                    <>
                        <motion.div
                            className="sidebar-overlay"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={overlayTransition}
                            onClick={() => setSidebarOpen(false)}
                        />
                        <motion.aside
                            className="sidebar-shell"
                            initial={{
                                x: -280,
                                opacity: 0
                            }}
                            animate={{
                                x: 0,
                                opacity: 1
                            }}
                            exit={{
                                x: -280,
                                opacity: 0
                            }}
                            transition={sidebarTransition}
                        >
                            <ThreadSidebar
                                threads={threads}
                                activeThreadId={activeThreadId}
                                onSelectThread={setActiveThreadId}
                                onAddThread={addThread}
                                onDeleteThread={deleteThread}
                                onRenameThread={renameThread}
                            />
                        </motion.aside>
                        <motion.button
                            className="sidebar-edge-toggle"
                            initial={{
                                x: -40,
                                opacity: 0
                            }}
                            animate={{
                                x: 280,
                                opacity: 1
                            }}
                            exit={{
                                x: -40,
                                opacity: 0
                            }}
                            transition={sidebarTransition}
                            onClick={() => setSidebarOpen(false)}
                            aria-label="Close Sidebar"
                        >
                            ➜
                        </motion.button>
                    </>
                )}
            </AnimatePresence>
            <div className="status-banner">
                Ved ready —
                <span className="highlight-mode">
                    {" "}
                    {currentMode.toUpperCase()} mode.
                </span>
            </div>
            <div className="output-viewport">
                {activeMessages.length === 0 ? (
                    <div className="empty-state">
                        No terminal messages log parsed.
                        Initialize engine prompt below.
                    </div>
                ) : (
                    activeMessages.map((msg, index) => (
                        <div
                            key={index}
                            className={`message-row ${msg.sender}-row`}
                        >
                            <div className="message-bubble">
                                <span className="sender-tag">
                                    {msg.sender === "user"
                                        ? "↳ User"
                                        : "⚙️ Ved"}
                                </span>
                                <p className="message-text">
                                    {msg.text}
                                </p>
                            </div>
                        </div>
                    ))
                )}
            </div>
            <ChatInput onSendMessage={sendMessage} />
        </div>
    );
};
export default Dashboard;