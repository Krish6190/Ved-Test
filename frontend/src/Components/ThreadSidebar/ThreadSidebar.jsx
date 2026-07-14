import React, { useState } from "react";
import "./ThreadSidebar.css";

const ThreadSidebar = ({
    threads,
    activeThreadId,
    onSelectThread,
    onAddThread,
    onDeleteThread,
    onRenameThread,
}) => {
    const [editingId, setEditingId] = useState(null);
    const [editTitle, setEditTitle] = useState("");

    const startEditing = (thread) => {
        setEditingId(thread.id);
        setEditTitle(thread.name);
    };

    const saveRename = (id) => {
        if (editTitle.trim()) {
            onRenameThread(id, editTitle.trim());
        }

        setEditingId(null);
    };

    return (
        <>
            <div className="sidebar-header">
                <h3>💬 Chat Threads</h3>

                <button
                    className="add-thread-btn"
                    onClick={onAddThread}
                    title="New Thread"
                >
                    ➕ New
                </button>
            </div>

            <div className="threads-list">
                {threads.map((thread) => (
                    <div
                        key={thread.id}
                        className={`thread-item ${
                            thread.id === activeThreadId ? "active" : ""
                        }`}
                        onClick={() => onSelectThread(thread.id)}
                    >
                        {editingId === thread.id ? (
                            <input
                                className="rename-input"
                                value={editTitle}
                                onChange={(e) =>
                                    setEditTitle(e.target.value)
                                }
                                onBlur={() => saveRename(thread.id)}
                                onKeyDown={(e) => {
                                    if (e.key === "Enter") {
                                        saveRename(thread.id);
                                    }
                                }}
                                autoFocus
                            />
                        ) : (
                            <span
                                className="thread-name"
                                onDoubleClick={() =>
                                    startEditing(thread)
                                }
                            >
                                {thread.name}
                            </span>
                        )}

                        <div className="thread-actions">
                            <button
                                className="action-sub-btn"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    startEditing(thread);
                                }}
                            >
                                ✏️
                            </button>

                            {threads.length > 1 && (
                                <button
                                    className="action-sub-btn del-btn"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onDeleteThread(thread.id);
                                    }}
                                >
                                    🗑️
                                </button>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </>
    );
};

export default ThreadSidebar;