const fs = require("fs");
const path = require("path");
const STATE_FILE = path.join(__dirname, "window-state.json");

function loadWindowState(defaults) {
    try {
        if (fs.existsSync(STATE_FILE)) {
            const saved = JSON.parse(
                fs.readFileSync(STATE_FILE, "utf8")
            );
            return {
                ...defaults,
                ...saved
            };
        }
    }
    catch {
    }
    return defaults;
}

function saveWindowState(window) {
    if (!window)
        return;
    const bounds = window.getBounds();
    fs.writeFileSync(
        STATE_FILE,
        JSON.stringify(bounds, null, 4)
    );
}

module.exports = {
    loadWindowState,
    saveWindowState
};