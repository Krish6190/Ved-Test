import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { Assets } from "../../constants/assets";
import CodeBlock from "./CodeBlock";
import "./Message.css";
const AssistantMessage = ({ message }) => {

    return (
        <div className="message-row assistant-row">
            <div className="message-card assistant-card">
                <div className="message-header">
                    <img src={Assets.logo} className="message-avatar" alt="VED" />
                    <span className="message-author">
                        VED
                    </span>
                </div>
                <div className="message-body">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                        components={{
                            code(props) {
                                const { inline, className, children } = props;
                                const match = /language-(\w+)/.exec(className || "");
                                if (!inline && match) {
                                    return (
                                        <CodeBlock
                                            language={match[1]}
                                        >
                                            {String(children).replace(/\n$/, "")}
                                        </CodeBlock>
                                    );
                                }
                                return (
                                    <code className={className}>
                                        {children}
                                    </code>
                                );
                            }
                        }}
                    >
                        {message.text}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
    );
};
export default AssistantMessage;