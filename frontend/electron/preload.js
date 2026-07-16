const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("ved", {

    close: () => ipcRenderer.invoke("window:close"),

    minimize: () => ipcRenderer.invoke("window:minimize"),

    restore: () => ipcRenderer.invoke("window:restore"),

    maximize: () => ipcRenderer.invoke("window:maximize"),

    toggleMaximize: () => ipcRenderer.invoke("window:toggleMaximize"),

    isMaximized: () => ipcRenderer.invoke("window:isMaximized"),

    setAlwaysOnTop: (value) =>
        ipcRenderer.invoke("window:setAlwaysOnTop", value),

    getVersion: () =>
        ipcRenderer.invoke("app:getVersion")

});