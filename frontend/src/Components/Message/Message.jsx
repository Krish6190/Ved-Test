import UserMessage from "./UserMessage";
import AssistantMessage from "./AssistantMessage";
import ToolMessage from "./ToolMessage";

const Message = ({ message }) => {

    switch (message.sender) {

        case "user":
            return <UserMessage message={message} />;

        case "tool":
            return <ToolMessage message={message} />;

        default:
            return <AssistantMessage message={message} />;

    }

};

export default Message;