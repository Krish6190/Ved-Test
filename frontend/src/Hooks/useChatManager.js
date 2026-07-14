import { useState } from 'react';

export const useChatManager = () => {
  const [threads, setThreads] = useState([
    { id: 1, name: 'New Session Thread', messages: [], isFresh: true }
  ]);
  const [activeThreadId, setActiveThreadId] = useState(1);

  const activeThread = threads.find(t => t.id === activeThreadId);

  const addThread = () => {
    const newId = Date.now();
    setThreads([...threads, { id: newId, name: 'New Session Thread', messages: [], isFresh: true }]);
    setActiveThreadId(newId);
  };

  const deleteThread = (id) => {
    const updatedThreads = threads.filter(t => t.id !== id);
    setThreads(updatedThreads);
    if (activeThreadId === id && updatedThreads.length > 0) {
      setActiveThreadId(updatedThreads[0].id);
    }
  };

  const renameThread = (id, newName) => {
    setThreads(threads.map(t => t.id === id ? { ...t, name: newName } : t));
  };

  const sendMessage = (text) => {
    setThreads(prevThreads => 
      prevThreads.map(thread => {
        if (thread.id !== activeThreadId) return thread;

        const updatedMessages = [...thread.messages, { sender: 'user', text }];
        let updatedName = thread.name;

        // Auto-summarizes the thread title using the first message string
        if (thread.isFresh) {
          const words = text.split(' ');
          updatedName = words.slice(0, 3).join(' ') + (words.length > 3 ? '...' : '');
        }

        return {
          ...thread,
          name: updatedName,
          messages: updatedMessages,
          isFresh: false
        };
      })
    );

    // Mock automated reply response block simulating background AI response
    setTimeout(() => {
      setThreads(prevThreads => 
        prevThreads.map(thread => {
          if (thread.id !== activeThreadId) return thread;
          return {
            ...thread,
            messages: [...thread.messages, { sender: 'ved', text: `Processed query execution context on standard parameters.` }]
          };
        })
      );
    }, 1000);
  };

  return {
    threads,
    activeThreadId,
    activeMessages: activeThread?.messages || [],
    addThread,
    deleteThread,
    renameThread,
    sendMessage,
    setActiveThreadId
  };
};
