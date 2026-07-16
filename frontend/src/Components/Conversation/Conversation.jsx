import { useEffect, useRef } from "react";
import Message from "../Message";
import "./Conversation.css";

const Conversation = ({ messages }) => {
    const bottomRef = useRef(null);
    useEffect(() => {
        bottomRef.current?.scrollIntoView({
            behavior: "smooth"
        });
    }, [messages]);

    return (
        <div className="conversation">
            {messages.length === 0 ? (
                <div className="conversation-empty">
                    <h2>Welcome to VED</h2>
                    <p>Start a conversation.</p>
                </div>
            ) : (
                <>
                    {messages.map((message, index) => (
                        <Message
                            key={message.id ?? index}
                            message={message}
                        />
                    ))}
                    <div ref={bottomRef} />
                </>
            )}
        </div>
    );
};
export default Conversation;