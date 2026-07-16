import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const CodeBlock = ({ language, children }) => {

    const copyCode = () => {
        navigator.clipboard.writeText(children);
    };

    return (
        <div className="code-block">

            <div className="code-header">

                <span>{language || "text"}</span>

                <button
                    className="copy-btn"
                    onClick={copyCode}
                >
                    Copy
                </button>

            </div>

            <SyntaxHighlighter
                language={language}
                style={oneDark}
                customStyle={{
                    margin: 0,
                    borderRadius: 0,
                    background: "#141821"
                }}
            >
                {children}
            </SyntaxHighlighter>

        </div>
    );

};

export default CodeBlock;