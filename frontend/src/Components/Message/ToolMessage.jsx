import "./Message.css";

const ToolMessage = ({ message }) => {

    return (

        <div className="message-row assistant-row">

            <div className="message-card tool-card">

                <div className="message-header">

                    <span className="message-avatar">
                        ⚙
                    </span>

                    <span className="message-author">
                        Tool
                    </span>

                </div>

                <div className="message-body">

                    {message.text}

                </div>

            </div>

        </div>

    );

};

export default ToolMessage;