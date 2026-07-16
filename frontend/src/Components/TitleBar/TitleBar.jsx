import "./TitleBar.css";
import { Assets } from "../../constants/assets";

const TitleBar = ({ currentPath }) => {
    return (
        <header className="title-bar">

            <div className="title-left">

                <img
                    src={Assets.logo}
                    alt="VED"
                    className="title-logo"
                />

                <div className="title-info">

                    <div className="title-name-row">

                        <span className="app-name">
                            VED Assistant
                        </span>

                        <span className="connection-status">
                            ● Ready
                        </span>

                    </div>

                    <div className="workspace-label">
                        Workspace
                    </div>

                    <div
                        className="directory-path"
                        title={currentPath}
                    >
                        {currentPath}
                    </div>

                </div>

            </div>

            <div className="window-controls">

                <button
                    className="control-btn"
                    onClick={() => window.ved.minimize()}
                    aria-label="Minimize"
                >
                    ─
                </button>

                <button
                    className="control-btn close-btn"
                    onClick={() => window.ved.close()}
                    aria-label="Close"
                >
                    ✕
                </button>

            </div>

        </header>
    );
};

export default TitleBar;