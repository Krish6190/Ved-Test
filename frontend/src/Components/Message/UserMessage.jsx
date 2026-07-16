import "./Message.css";

const UserMessage = ({ message }) => {

    return (

        <div className="message-row user-row">

            <div className="message-card user-card">

                <div className="message-header">

                    <span className="message-avatar">
                        👤
                    </span>

                    <span className="message-author">
                        You
                    </span>

                </div>

                <div className="message-body">

                    {message.text}

                </div>

            </div>

        </div>

    );

};

export default UserMessage;