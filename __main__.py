import os
from ui.gui import main
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if __name__ == "__main__":
    main()
