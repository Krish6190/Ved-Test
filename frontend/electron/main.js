const { app, BrowserWindow, ipcMain, screen, Tray, Menu } = require("electron");
const path = require('path');
let mainWindow;
let tray = null;
const {
  loadWindowState,
  saveWindowState
} = require("./windowState");

function createWindow() {
  const defaultState = {
    width: 560,
    height: 760,
    x: undefined,
    y: undefined
  };
  const windowState = loadWindowState(defaultState);
  mainWindow = new BrowserWindow({
    ...windowState,
    frame: false,
    minWidth: 480,
    minHeight: 620,
    resizable: true,
    roundedCorners: true,
    backgroundColor: "#0b0b0b",
    frame: false, // Disables standard OS border window bar decorations to use your custom TitleBar UI
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true
    }
  });
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  if (windowState.x === undefined || windowState.y === undefined) {
    const { width, height } =
      screen.getPrimaryDisplay().workAreaSize;
    mainWindow.setPosition(
      width - 580,
      height - 800
    );
  }
  // If in development mode, load Vite server port. In production, look at dist index bundle build file.
  const startUrl = process.env.ELECTRON_START_URL || `file://${path.join(__dirname, '../dist/index.html')}`;
  mainWindow.loadURL(startUrl);
  mainWindow.on("close", (event) => {
    if (!app.isQuiting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  mainWindow.on("move", () => {
    saveWindowState(mainWindow);
  });
  mainWindow.on("resize", () => {
    saveWindowState(mainWindow);
  });
}

app.whenReady().then(() => {
  createWindow();
  const fs = require("fs");
  const trayPath = path.join(__dirname, "assets", "ved.png");
  console.log("Tray path:", trayPath);
  console.log("Exists:", fs.existsSync(trayPath));
  tray = new Tray(trayPath);
  tray.setToolTip("VED Assistant");
  const trayMenu = Menu.buildFromTemplate([
    {
      label: "Open VED", click() { if (!mainWindow) return; mainWindow.show(); mainWindow.focus(); }
    },
    {
      type: "separator"
    },
    {
      label: "Exit", click() { app.isQuiting = true; app.quit(); }
    }
  ]);
  tray.setContextMenu(trayMenu);
  tray.on("click", () => {
    if (!mainWindow) return;
    if (mainWindow.isVisible()) mainWindow.hide();
    else {
      mainWindow.show();
      mainWindow.focus();
    }
  });
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

ipcMain.handle("window:close", () => {
  mainWindow.close();
});

ipcMain.handle("window:minimize", () => {
  mainWindow.minimize();
});

ipcMain.handle("window:restore", () => {
  mainWindow.restore();
});

ipcMain.handle("window:maximize", () => {
  mainWindow.maximize();
});

ipcMain.handle("window:toggleMaximize", () => {
  if (mainWindow.isMaximized())
    mainWindow.unmaximize();
  else
    mainWindow.maximize();
});

ipcMain.handle("window:isMaximized", () => {
  return mainWindow.isMaximized();
});

ipcMain.handle("window:setAlwaysOnTop", (_, value) => {
  mainWindow.setAlwaysOnTop(value);
});

ipcMain.handle("app:getVersion", () => {
  return app.getVersion();
});